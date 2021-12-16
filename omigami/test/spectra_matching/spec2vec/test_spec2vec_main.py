from unittest.mock import Mock

import pytest

import omigami.spectra_matching.spec2vec.main
from omigami.authentication.prefect_factory import prefect_client_factory
from omigami.config import (
    SOURCE_URI_PARTIAL_GNPS,
)
from omigami.deployer import FlowDeployer
from omigami.spectra_matching.spec2vec.factory import Spec2VecFlowFactory
from omigami.spectra_matching.spec2vec.main import (
    run_spec2vec_training_flow,
    run_deploy_spec2vec_model_flow,
)


@pytest.fixture
def mock_objects(monkeypatch):
    client_factory = prefect_client_factory
    client_factory.get = Mock()
    monkeypatch.setattr(
        omigami.spectra_matching.spec2vec.main,
        "prefect_client_factory",
        client_factory,
    )

    mock_deployer = Mock(spec=FlowDeployer)
    deployer_instance = mock_deployer.return_value
    monkeypatch.setattr(
        omigami.spectra_matching.spec2vec.main, "FlowDeployer", mock_deployer
    )

    mock_flow_factory = Mock(spec=Spec2VecFlowFactory)
    monkeypatch.setattr(
        omigami.spectra_matching.spec2vec.main, "Spec2VecFlowFactory", mock_flow_factory
    )
    factory_instance = mock_flow_factory.return_value

    return {
        "deployer": deployer_instance,
        "client": client_factory,
        "factory": factory_instance,
    }


def test_mocked_deploy_training_flow(mock_objects):
    mock_objects["factory"].build_training_flow = Mock(return_value="flow")
    mock_objects["deployer"].deploy_flow = Mock(return_value=("id", "run_id"))

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
        schedule=None,
    )

    flow_id, flow_run_id = run_spec2vec_training_flow(**params)

    assert (flow_id, flow_run_id) == ("id", "run_id")
    mock_objects["factory"].build_training_flow.assert_called_once_with(**params)
    mock_objects["deployer"].deploy_flow.assert_called_once_with(
        flow="flow", project_name="default"
    )
    mock_objects["client"].get.assert_called_once()


def test_run_mocked_deploy_model_flow(mock_objects):
    mock_objects["factory"].build_model_deployment_flow = Mock(return_value="flow")
    mock_objects["deployer"].deploy_flow = Mock(return_value=("id", "run_id"))

    params = dict(
        flow_name="Calvin Flowers",
        image="star wars episode II wasn't so bad",
        intensity_weighting_power=0.4,
        allowed_missing_percentage=10,
        dataset_id="small",
        n_decimals=2,
        ion_mode="positive",
        project_name="default",
    )

    flow_id, flow_run_id = run_deploy_spec2vec_model_flow(
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
