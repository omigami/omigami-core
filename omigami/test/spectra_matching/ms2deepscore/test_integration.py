import pytest

from omigami.config import STORAGE_ROOT, SOURCE_URI_PARTIAL_GNPS_500_SPECTRA
from omigami.spectra_matching.ms2deepscore.main import run_ms2deepscore_flow
from omigami.spectra_matching.storage.model_registry import MLFlowDataGateway
from omigami.test.spectra_matching.conftest import monitor_flow_results


@pytest.mark.skip(
    "Requires local prefect server and mlflow. Make sure they are running to run this"
    "test. To run them, check README instructions. When running the first time, the"
    "dataset will be downloaded to local-deployment/ - for the first run it needs"
    "internet connection."
)
def test_deploy_training_flow(backend_services):
    client = backend_services["prefect"]

    flow_id, flow_run_id = run_ms2deepscore_flow(
        image="",
        project_name="local-integration-test-ms2ds",
        flow_name="MS2DS Flow",
        dataset_id="small",
        dataset_directory=str(STORAGE_ROOT.parent / "datasets-ms2ds"),
        source_uri=SOURCE_URI_PARTIAL_GNPS_500_SPECTRA,
        ion_mode="positive",
        deploy_model=False,
        overwrite_model=False,
        overwrite_all_spectra=True,
        schedule=None,
        spectrum_ids_chunk_size=100,
        fingerprint_n_bits=2048,
        scores_decimals=5,
        spectrum_binner_n_bins=10000,
        train_ratio=0.8,
        validation_ratio=0.2,
        test_ratio=0.2,
        epochs=5,
        chunk_size=150000,
    )

    monitor_flow_results(client, flow_run_id)
    assert client.get_flow_run_state(flow_run_id).is_successful()
