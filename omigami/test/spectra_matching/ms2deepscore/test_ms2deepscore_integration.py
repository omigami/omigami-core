import pytest

from omigami.config import (
    STORAGE_ROOT,
)
from omigami.spectra_matching.ms2deepscore.main import (
    run_ms2deepscore_training_flow,
    run_deploy_ms2ds_model_flow,
)
from omigami.test.spectra_matching.conftest import monitor_flow_results


@pytest.mark.skip(
    "Requires local prefect server and mlflow. Make sure they are running to run this"
    "test. To run them, check README instructions. When running the first time, the"
    "dataset will be downloaded to local-deployment/ - for the first run it needs"
    "internet connection."
)
def test_run_training_flow(backend_services):
    client = backend_services["prefect"]

    flow_id, flow_run_id = run_ms2deepscore_training_flow(
        image="",
        project_name="local-integration-test-ms2ds",
        flow_name="MS2DS Flow",
        dataset_id="small",
        dataset_directory=str(STORAGE_ROOT.parent / "datasets-ms2ds"),
        ion_mode="positive",
        deploy_model=False,
        overwrite_model=False,
        schedule=None,
        spectrum_ids_chunk_size=100,
        fingerprint_n_bits=2048,
        scores_decimals=5,
        spectrum_binner_n_bins=10000,
        train_ratio=0.8,
        validation_ratio=0.2,
        test_ratio=0.2,
        epochs=5,
    )

    monitor_flow_results(client, flow_run_id)
    assert client.get_flow_run_state(flow_run_id).is_successful()


@pytest.mark.skip(
    "Requires local prefect server and mlflow. Make sure they are running to run this"
    "test. To run them, check README instructions."
)
def test_run_model_deployment_flow(
    backend_services, registered_ms2ds_model, mock_ms2ds_deploy_model_task
):
    client = backend_services["prefect"]

    flow_id, flow_run_id = run_deploy_ms2ds_model_flow(
        model_run_id=registered_ms2ds_model["run_id"],
        image="",
        project_name="local-integration-test-ms2ds",
        flow_name="MS2DS - Model Deployment Flow",
        dataset_id="small",
        ion_mode="positive",
    )

    monitor_flow_results(client, flow_run_id)
    assert client.get_flow_run_state(flow_run_id).is_successful()
