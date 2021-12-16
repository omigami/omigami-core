from unittest.mock import Mock

import pytest

import omigami.spectra_matching.ms2deepscore.main
from omigami.authentication.prefect_factory import prefect_client_factory
from omigami.config import (
    SOURCE_URI_PARTIAL_GNPS_500_SPECTRA,
)
from omigami.deployer import FlowDeployer
from omigami.spectra_matching.ms2deepscore.factory import MS2DeepScoreFlowFactory
from omigami.spectra_matching.ms2deepscore.main import (
    run_ms2deepscore_training_flow,
    run_deploy_ms2ds_model_flow,
)


@pytest.fixture
def mock_objects(monkeypatch):
    client_factory = prefect_client_factory
    client_factory.get = Mock()
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.main,
        "prefect_client_factory",
        client_factory,
    )

    mock_deployer = Mock(spec=FlowDeployer)
    deployer_instance = mock_deployer.return_value
    deployer_instance.deploy_flow = Mock(return_value=("id", "run_id"))
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.main, "FlowDeployer", mock_deployer
    )

    mock_flow_factory = Mock(spec=MS2DeepScoreFlowFactory)
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.main,
        "MS2DeepScoreFlowFactory",
        mock_flow_factory,
    )
    factory_instance = mock_flow_factory.return_value

    return {
        "deployer": deployer_instance,
        "client": client_factory,
        "factory": factory_instance,
    }


def test_mocked_deploy_training_flow(mock_objects):
    mock_objects["factory"].build_training_flow = Mock(return_value="flow")

    params = dict(
        image="",
        project_name="default",
        flow_name="Robert DeFlow",
        dataset_id="small",
        source_uri=SOURCE_URI_PARTIAL_GNPS_500_SPECTRA,
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

    flow_id, flow_run_id = run_ms2deepscore_training_flow(**params)

    assert (flow_id, flow_run_id) == ("id", "run_id")
    mock_objects["factory"].build_training_flow.assert_called_once_with(**params)
    mock_objects["deployer"].deploy_flow.assert_called_once_with(
        flow="flow", project_name="default"
    )
    mock_objects["client"].get.assert_called_once()


def test_run_mocked_deploy_model_flow(mock_objects):
    mock_objects["factory"].build_model_deployment_flow = Mock(return_value="flow")

    params = dict(
        flow_name="Calvin Flowers",
        image="star wars episode II wasn't so bad",
        dataset_id="small",
        ion_mode="positive",
        project_name="default",
    )

    flow_id, flow_run_id = run_deploy_ms2ds_model_flow(
        model_run_id="model_run_id", **params
    )

    assert (flow_id, flow_run_id) == ("id", "run_id")
    mock_objects["factory"].build_model_deployment_flow.assert_called_once_with(
        **params
    )
    mock_objects["deployer"].deploy_flow.assert_called_once_with(
        flow="flow",
        project_name="default",
        flow_parameters={"ModelRunID": "model_run_id"},
    )
    mock_objects["client"].get.assert_called_once()
