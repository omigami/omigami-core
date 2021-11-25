from unittest.mock import Mock

import pytest
from prefect import Flow
from prefect.run_configs import LocalRun
from prefect.storage import Local

import omigami.spec2vec.main
from omigami.config import (
    SOURCE_URI_PARTIAL_GNPS,
    STORAGE_ROOT,
    DATASET_IDS,
)
from omigami.deployer import FlowDeployer
from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.factory import Spec2VecFlowFactory
from omigami.spec2vec.main import run_spec2vec_flow
from omigami.tasks import DownloadData, DownloadParameters
from omigami.test.conftest import monitor_flow_results


@pytest.mark.skip(
    "Requires local prefect server and mlflow. Make sure they are running to run this"
    "test. To run them, check README instructions."
)
def test_deploy_training_flow(backend_services):
    client = backend_services["prefect"]

    flow_id, flow_run_id = run_spec2vec_flow(
        image="",
        project_name="local-integration-test-s2v",
        flow_name="Robert DeFlow",
        dataset_id="small",
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
        dataset_directory=STORAGE_ROOT.parent / "datasets",
    )

    monitor_flow_results(client, flow_run_id)
    assert client.get_flow_run_state(flow_run_id).is_successful()


@pytest.mark.skip(
    "Requires local prefect server and mlflow. Check readme to start them."
)
def test_single_task_local_integration(backend_services):
    """This is covered by the integration test above but was used for development
    and it is useful for debugging"""
    client = backend_services["prefect"]
    params = DownloadParameters(
        SOURCE_URI_PARTIAL_GNPS, str(STORAGE_ROOT), DATASET_IDS["small"]
    )
    with Flow(
        "test-flow",
        storage=Local(str(STORAGE_ROOT)),
        run_config=LocalRun(working_dir=STORAGE_ROOT, labels=["dev"]),
    ) as flow:
        _ = DownloadData(
            FSDataGateway(),
            params,
        )()

    client.create_project("default")
    flow_id = client.register(flow, project_name="default")
    flow_run_id = client.create_flow_run(flow_id=flow_id, run_name=f"test run")

    monitor_flow_results(client, flow_run_id)
    assert client.get_flow_run_state(flow_run_id).is_successful()


def test_mocked_deploy_training_flow(monkeypatch):
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

    mock_deployer = Mock(spec=FlowDeployer)
    deployer_instance = mock_deployer.return_value
    deployer_instance.deploy_flow = Mock(return_value=("id", "run_id"))
    monkeypatch.setattr(omigami.spec2vec.main, "FlowDeployer", mock_deployer)

    params = dict(
        image="",
        project_name="default",
        flow_name="Robert DeFlow",
        dataset_id="small",
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

    flow_id, flow_run_id = run_spec2vec_flow(**params)

    assert (flow_id, flow_run_id) == ("id", "run_id")
    mock_client_factory.assert_called_once()
    client_factory_instance.get_client.assert_called_once()
    mock_flow_factory.assert_called_once()
    factory_instance.build_training_flow.assert_called_once_with(**params)
    mock_deployer.assert_called_once_with(prefect_client="client")
    deployer_instance.deploy_flow.assert_called_once_with(
        flow="flow", project_name="default"
    )
