from unittest.mock import Mock

import omigami.spectra_matching.ms2deepscore.main
from omigami.config import (
    SOURCE_URI_PARTIAL_GNPS_500_SPECTRA,
)
from omigami.deployer import FlowDeployer
from omigami.spectra_matching.ms2deepscore.factory import MS2DeepScoreFlowFactory
from omigami.spectra_matching.ms2deepscore.main import run_ms2deepscore_flow


def test_mocked_deploy_training_flow(monkeypatch):
    mock_client_factory = Mock()
    client_factory_instance = mock_client_factory.return_value
    client_factory_instance.get_client = Mock(return_value="client")
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.main,
        "PrefectClientFactory",
        mock_client_factory,
    )

    mock_flow_factory = Mock(spec=MS2DeepScoreFlowFactory)
    factory_instance = mock_flow_factory.return_value
    factory_instance.build_training_flow = Mock(return_value="flow")
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.main,
        "MS2DeepScoreFlowFactory",
        mock_flow_factory,
    )

    mock_deployer = Mock(spec=FlowDeployer)
    deployer_instance = mock_deployer.return_value
    deployer_instance.deploy_flow = Mock(return_value=("id", "run_id"))
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.main, "FlowDeployer", mock_deployer
    )

    params = dict(
        image="",
        project_name="default",
        flow_name="Robert DeFlow",
        dataset_id="small",
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

    flow_id, flow_run_id = run_ms2deepscore_flow(**params)

    assert (flow_id, flow_run_id) == ("id", "run_id")
    mock_client_factory.assert_called_once()
    client_factory_instance.get_client.assert_called_once()
    mock_flow_factory.assert_called_once()
    factory_instance.build_training_flow.assert_called_once_with(**params)
    mock_deployer.assert_called_once_with(prefect_client="client")
    deployer_instance.deploy_flow.assert_called_once_with(
        flow="flow", project_name="default"
    )
