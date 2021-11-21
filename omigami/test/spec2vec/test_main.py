from time import sleep
from unittest.mock import Mock

import mlflow
import pytest
from prefect import Flow
from prefect.run_configs import LocalRun
from prefect.storage import Local

import omigami.spec2vec.main
from omigami.authentication.prefect_factory import PrefectClientFactory
from omigami.config import (
    SOURCE_URI_PARTIAL_GNPS,
    STORAGE_ROOT,
    DATASET_IDS,
    config,
    API_SERVER_URLS,
)
from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.deployment import Spec2VecDeployer
from omigami.spec2vec.factory import Spec2VecFlowFactory
from omigami.spec2vec.main import run_training_flow
from omigami.tasks import DownloadData, DownloadParameters


@pytest.fixture()
def backend_services():
    mlflow.set_tracking_uri("mysql+pymysql://root:password123@127.0.0.1:3306/mlflow")
    mlflow_client = mlflow.tracking.MlflowClient()

    login_config = config["login"]["local"].get(dict)
    api_server = API_SERVER_URLS["local"]
    prefect_factory = PrefectClientFactory(api_server=api_server, **login_config)
    prefect_client = prefect_factory.get_client()

    if not prefect_client.active_tenant_id:
        prefect_client.create_tenant("default")

    return {"prefect": prefect_client, "mlflow": mlflow_client}


# @pytest.mark.skip(
#     "Requires local prefect server and mlflow. Make sure they are running to run this
#     test. To run them, check README instructions."
# )
def test_run_training_flow(backend_services):
    client = backend_services["prefect"]

    flow_id, flow_run_id = run_training_flow(
        image="",
        project_name="default",
        flow_name="Robert DeFlow",
        dataset_name="small",
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        ion_mode="positive",
        iterations=3,
        n_decimals=2,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=15,
        environment="local",
        deploy_model=False,
        overwrite_model=False,
        overwrite_all_spectra=True,
        schedule=None,
        auth=False,
    )

    while (
        not client.get_flow_run_state(flow_run_id).is_successful()
        or client.get_flow_run_state(flow_run_id).is_failed()
    ):
        sleep(0.5)

    assert client.get_flow_run_state(flow_run_id).is_successful()


@pytest.mark.skip(
    "Requires local prefect server and mlflow. Check readme to start them."
)
def test_single_task_local_integration(backend_services):
    client = backend_services["prefect"]
    params = DownloadParameters(
        SOURCE_URI_PARTIAL_GNPS, str(STORAGE_ROOT), DATASET_IDS["small"]
    )
    with Flow(
        "test-flow",
        storage=Local(str(STORAGE_ROOT)),
        run_config=LocalRun(working_dir=STORAGE_ROOT, labels=["dev"]),
    ) as flow:
        res = DownloadData(
            FSDataGateway(),
            params,
        )()

    client.create_project("default")
    flow_id = client.register(flow, project_name="default")

    flow_run_id = client.create_flow_run(flow_id=flow_id, run_name=f"test run")
    while (
        not client.get_flow_run_state(flow_run_id).is_successful()
        or client.get_flow_run_state(flow_run_id).is_failed()
    ):
        sleep(0.5)

    assert client.get_flow_run_state(flow_run_id).is_successful()


def test_mocked_run_training_flow(monkeypatch):
    mock_client_factory = Mock()
    client_factory_instance = mock_client_factory.return_value
    client_factory_instance.get_client = Mock(return_value="client")
    monkeypatch.setattr(
        omigami.spec2vec.main,
        "PrefectClientFactory",
        mock_client_factory,
    )

    mock_flow_factory = Mock(spec=Spec2VecFlowFactory)
    factory_instance = mock_flow_factory.return_value
    factory_instance.build_training_flow = Mock(return_value="flow")
    monkeypatch.setattr(omigami.spec2vec.main, "Spec2VecFlowFactory", mock_flow_factory)

    mock_deployer = Mock(spec=Spec2VecDeployer)
    deployer_instance = mock_deployer.return_value
    deployer_instance.deploy_flow = Mock(return_value=("id", "run_id"))
    monkeypatch.setattr(omigami.spec2vec.main, "Spec2VecDeployer", mock_deployer)

    flow_id, flow_run_id = run_training_flow(
        image="",
        project_name="default",
        flow_name="Robert DeFlow",
        dataset_name="small",
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        ion_mode="positive",
        iterations=3,
        n_decimals=2,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=15,
        environment="local",
        deploy_model=False,
        overwrite_model=False,
        overwrite_all_spectra=True,
        schedule=None,
        auth=False,
    )

    assert (flow_id, flow_run_id) == ("id", "run_id")
    mock_client_factory.assert_called_once()
    client_factory_instance.get_client.assert_called_once()
    mock_flow_factory.assert_called_once_with(environment="local")
    factory_instance.build_training_flow.assert_called_once_with(
        image="",
        project_name="default",
        flow_name="Robert DeFlow",
        dataset_name="small",
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        ion_mode="positive",
        iterations=3,
        n_decimals=2,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=15,
        deploy_model=False,
        overwrite_model=False,
        overwrite_all_spectra=True,
        schedule=None,
    )
    mock_deployer.assert_called_once_with(client="client")
    deployer_instance.deploy_flow.assert_called_once_with(
        flow="flow", project_name="default"
    )