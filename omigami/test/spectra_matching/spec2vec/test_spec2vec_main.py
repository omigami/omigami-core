from unittest.mock import Mock

import pytest

import omigami.spectra_matching.spec2vec.main
from omigami.config import (
    SOURCE_URI_PARTIAL_GNPS,
)
from omigami.deployer import FlowDeployer
from omigami.spectra_matching.spec2vec.factory import Spec2VecFlowFactory
from omigami.spectra_matching.spec2vec.main import (
    run_spec2vec_flow,
    run_deploy_model_flow,
)


@pytest.fixture
def mock_factories(monkeypatch):
    get_client = Mock()
    monkeypatch.setattr(
        omigami.spectra_matching.spec2vec.main,
        "_get_prefect_client",
        get_client,
    )

    mock_deployer = Mock(spec=FlowDeployer)
    deployer_instance = mock_deployer.return_value
    deployer_instance.deploy_flow = Mock(return_value=("id", "run_id"))
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
        "get_client": get_client,
        "factory": factory_instance,
    }


def test_mocked_deploy_training_flow(mock_factories):
    mock_factories["factory"].build_spec2vec_flow = Mock(return_value="flow")

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
    mock_factories["factory"].build_spec2vec_flow.assert_called_once_with(**params)
    mock_factories["deployer"].deploy_flow.assert_called_once_with(
        flow="flow", project_name="default"
    )
    mock_factories["get_client"].assert_called_once()


def test_run_mocked_deploy_model_flow(mock_factories):
    mock_factories["factory"].build_model_deployment_flow = Mock(return_value="flow")

    params = dict(
        flow_name="Calvin Flowers",
        image="star wars episode II wasn't so bad",
        intensity_weighting_power=0.4,
        allowed_missing_percentage=10,
        dataset_id="small",
        n_decimals=2,
        ion_mode="positive",
        overwrite_model=True,
        project_name="default",
    )

    flow_id, flow_run_id = run_deploy_model_flow(model_run_id="model_run_id", **params)

    assert (flow_id, flow_run_id) == ("id", "run_id")
    mock_factories["factory"].build_model_deployment_flow.assert_called_once_with(
        **params
    )
    mock_factories["deployer"].deploy_flow.assert_called_once_with(
        flow="flow", project_name="default", parameters={"ModelRunID": "model_run_id"}
    )
    mock_factories["get_client"].assert_called_once()
