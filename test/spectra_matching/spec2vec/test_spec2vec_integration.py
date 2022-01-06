import pytest
from prefect import Flow
from prefect.run_configs import LocalRun
from prefect.storage import Local

from omigami.config import (
    GNPS_URIS,
    STORAGE_ROOT,
    DATASET_IDS,
)
from omigami.spectra_matching.spec2vec.main import (
    run_spec2vec_training_flow,
    run_deploy_spec2vec_model_flow,
)
from omigami.spectra_matching.storage import FSDataGateway
from omigami.spectra_matching.tasks import DownloadParameters, DownloadData
from test.spectra_matching.conftest import monitor_flow_results


@pytest.mark.skip(
    "Requires local prefect server and mlflow. Make sure they are running to run this"
    "test. To run them, check README instructions."
)
def test_deploy_training_flow(backend_services, mock_s2v_deploy_model_task):
    client = backend_services["prefect"]
    image = None  # "drtools/omigami-spec2vec:SNAPSHOT.1a6ec12c"

    flow_id, flow_run_id = run_spec2vec_training_flow(
        image=image,
        project_name="local-integration-test-s2v",
        flow_name="Robert DeFlow",
        dataset_id="small",
        ion_mode="positive",
        iterations=3,
        n_decimals=1,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=15,
        deploy_model=False,
        overwrite_model=False,
        schedule=None,
        dataset_directory=STORAGE_ROOT.parent / "datasets",
    )

    monitor_flow_results(client, flow_run_id)
    assert client.get_flow_run_state(flow_run_id).is_successful()


@pytest.mark.skip(
    "Requires local prefect server and mlflow. Make sure they are running to run this"
    "test. To run them, check README instructions."
)
def test_download_task_local_integration(backend_services):
    """This is covered by the integration test above but was used for development
    and it is useful for debugging"""
    client = backend_services["prefect"]
    params = DownloadParameters(
        GNPS_URIS["small"], str(STORAGE_ROOT), DATASET_IDS["small"]
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


@pytest.mark.skip(
    "Requires local prefect server and mlflow. Make sure they are running to run this"
    "test. To run them, check README instructions."
)
def test_run_model_deployment_flow(
    backend_services,
    mock_s2v_deploy_model_task,
    word2vec_model,
    mock_default_config,
    registered_s2v_model,
):
    client = backend_services["prefect"]

    flow_id, flow_run_id = run_deploy_spec2vec_model_flow(
        model_run_id=registered_s2v_model["run_id"],
        image="",
        project_name="local-integration-test-s2v",
        flow_name="Model Deployment Flow",
        dataset_id="small",
        ion_mode="positive",
        n_decimals=1,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=15,
    )

    monitor_flow_results(client, flow_run_id)
    assert client.get_flow_run_state(flow_run_id).is_successful()
