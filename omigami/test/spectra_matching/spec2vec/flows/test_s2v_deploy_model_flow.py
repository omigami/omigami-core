import pytest

from omigami.spectra_matching.spec2vec.flows.deploy_model import (
    DeployModelFlowParameters,
    build_deploy_model_flow,
)
from omigami.spectra_matching.spec2vec.predictor import Spec2VecPredictor
from omigami.spectra_matching.storage import RedisSpectrumDataGateway, FSDataGateway
from omigami.spectra_matching.storage.model_registry import MLFlowDataGateway


def test_s2v_deploy_model_flow(flow_config):
    expected_tasks = {
        "ModelRunID",  # a prefect `Parameter` is actually a Task too
        "ListDocumentPaths",
        "MakeEmbeddings",
        "DeployModel",
        "LoadSpec2VecModel",
    }
    params = DeployModelFlowParameters(
        spectrum_dgw=RedisSpectrumDataGateway("project"),
        fs_dgw=FSDataGateway(),
        ion_mode="positive",
        n_decimals=2,
        documents_directory="directory",
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5.0,
        redis_db="0",
    )

    deploy_model_flow = build_deploy_model_flow("deploy-flow", flow_config, params)
    list_task = deploy_model_flow.get_tasks("ListDocumentPaths")[0]

    assert deploy_model_flow.name == "deploy-flow"
    assert list_task._documents_directory == "directory"
    task_names = {t.name for t in deploy_model_flow.tasks}
    assert task_names == expected_tasks


@pytest.fixture()
def deploy_model_setup(
    tmpdir_factory, word2vec_model, monkeypatch, mock_s2v_deploy_model_task
):
    tmpdir = tmpdir_factory.mktemp("model")
    mlflow_uri = f"sqlite:///{tmpdir}/mlflow.sqlite"
    dgw = MLFlowDataGateway(mlflow_uri)
    model = Spec2VecPredictor(
        word2vec_model,
        ion_mode="positive",
        n_decimals=2,
        intensity_weighting_power=1.0,
        allowed_missing_percentage=15,
    )
    run_id = dgw.register_model(
        model=model,
        run_name="run",
        experiment_name="test-experiment",
        model_name="test",
        experiment_path=str(tmpdir),
    )

    return {"run_id": run_id, "mlflow_uri": mlflow_uri}


def test_run_s2v_deploy_model_flow(
    deploy_model_setup,
    flow_config,
    mock_default_config,
    monkeypatch,
    spec2vec_redis_setup,
    s3_documents_directory,
):
    params = DeployModelFlowParameters(
        spectrum_dgw=RedisSpectrumDataGateway("project"),
        fs_dgw=FSDataGateway(),
        ion_mode="positive",
        n_decimals=1,
        documents_directory=s3_documents_directory,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5.0,
        redis_db="0",
        model_registry_uri=deploy_model_setup["mlflow_uri"],
    )

    deploy_model_flow = build_deploy_model_flow("deploy-flow", flow_config, params)

    flow_run = deploy_model_flow.run(
        parameters={"ModelRunID": deploy_model_setup["run_id"]}
    )

    assert flow_run.is_successful()
