from unittest.mock import Mock

import pytest

import omigami.spec2vec.main
from omigami.config import (
    SOURCE_URI_PARTIAL_GNPS_500_SPECTRA,
    STORAGE_ROOT,
)
from omigami.deployer import FlowDeployer
from omigami.ms2deepscore.factory import MS2DeepScoreFlowFactory
from omigami.ms2deepscore.main import run_ms2deepscore_flow
from omigami.test.conftest import monitor_flow_results


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
        dataset_directory=str(STORAGE_ROOT / "datasets-ms2ds"),
        source_uri=SOURCE_URI_PARTIAL_GNPS_500_SPECTRA,
        ion_mode="positive",
        deploy_model=False,
        overwrite_model=False,
        overwrite_all_spectra=True,
        schedule=None,
        authenticate=False,
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


def test_mocked_deploy_training_flow(monkeypatch):
    mock_client_factory = Mock()
    client_factory_instance = mock_client_factory.return_value
    client_factory_instance.get_client = Mock(return_value="client")
    monkeypatch.setattr(
        omigami.spec2vec.main,
        "PrefectClientFactory",
        mock_client_factory,
    )

    mock_flow_factory = Mock(spec=MS2DeepScoreFlowFactory)
    factory_instance = mock_flow_factory.return_value
    factory_instance.build_training_flow = Mock(return_value="flow")
    monkeypatch.setattr(
        omigami.ms2deepscore.main, "MS2DeepScoreFlowFactory", mock_flow_factory
    )

    mock_deployer = Mock(spec=FlowDeployer)
    deployer_instance = mock_deployer.return_value
    deployer_instance.deploy_flow = Mock(return_value=("id", "run_id"))
    monkeypatch.setattr(omigami.ms2deepscore.main, "FlowDeployer", mock_deployer)

    flow_id, flow_run_id = run_ms2deepscore_flow(
        image="",
        project_name="local-integration-test",
        flow_name="Robert DeFlow",
        dataset_id="small",
        source_uri=SOURCE_URI_PARTIAL_GNPS_500_SPECTRA,
        ion_mode="positive",
        deploy_model=False,
        overwrite_model=False,
        overwrite_all_spectra=True,
        schedule=None,
        authenticate=False,
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

    assert (flow_id, flow_run_id) == ("id", "run_id")
    mock_client_factory.assert_called_once()
    client_factory_instance.get_client.assert_called_once()
    mock_flow_factory.assert_called_once()
    factory_instance.build_training_flow.assert_called_once_with(
        image="",
        project_name="default",
        flow_name="Robert DeFlow",
        dataset_id="small",
        source_uri=SOURCE_URI_PARTIAL_GNPS_500_SPECTRA,
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
    mock_deployer.assert_called_once_with(prefect_client="client")
    deployer_instance.deploy_flow.assert_called_once_with(
        flow="flow", project_name="default"
    )
