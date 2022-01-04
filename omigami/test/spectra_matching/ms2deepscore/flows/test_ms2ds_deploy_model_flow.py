import pytest

from omigami.config import MLFLOW_DIRECTORY
from omigami.spectra_matching.ms2deepscore.flows.deploy_model import (
    DeployModelFlowParameters,
    build_deploy_model_flow,
)
from omigami.spectra_matching.ms2deepscore.storage import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.spectra_matching.ms2deepscore.tasks import (
    RegisterModelParameters,
    TrainModelParameters,
    RegisterModel,
)


def test_ms2ds_deploy_model_flow(flow_config):
    expected_tasks = {
        "ModelRunID",  # a prefect `Parameter` is actually a Task too
        "GetMS2DeepScoreModelPath",
        "MakeEmbeddings",
        "DeployModel",
        "CreateSpectrumIDsChunks",
        "DeleteEmbeddings",
    }
    params = DeployModelFlowParameters(
        spectrum_dgw=MS2DeepScoreRedisSpectrumDataGateway(),
        fs_dgw=MS2DeepScoreFSDataGateway(),
        ion_mode="positive",
        redis_db="0",
    )

    deploy_model_flow = build_deploy_model_flow("deploy-flow", flow_config, params)

    assert deploy_model_flow.name == "deploy-flow"
    task_names = {t.name for t in deploy_model_flow.tasks}
    assert task_names == expected_tasks


@pytest.fixture()
def deploy_model_setup(
    tmpdir_factory,
    mock_ms2ds_deploy_model_task,
    siamese_model_path,
):
    tmpdir = tmpdir_factory.mktemp("model")
    mlflow_uri = f"sqlite:///{tmpdir}/mlflow.sqlite"
    params = RegisterModelParameters(
        "test_experiment", mlflow_uri, str(MLFLOW_DIRECTORY), "positive"
    )
    train_params = TrainModelParameters("path", "positive", "path")
    register_task = RegisterModel(params, train_params)
    model_run_id = register_task.run(
        {"ms2deepscore_model_path": siamese_model_path, "validation_loss": 0.5}
    )

    return {"run_id": model_run_id, "mlflow_uri": mlflow_uri}


def test_run_ms2ds_deploy_model_flow(
    deploy_model_setup,
    flow_config,
    mock_default_config,
    redis_full_setup,
):
    params = DeployModelFlowParameters(
        spectrum_dgw=MS2DeepScoreRedisSpectrumDataGateway(),
        fs_dgw=MS2DeepScoreFSDataGateway(),
        ion_mode="positive",
        redis_db="0",
        model_registry_uri=deploy_model_setup["mlflow_uri"],
    )

    deploy_model_flow = build_deploy_model_flow("deploy-flow", flow_config, params)

    flow_run = deploy_model_flow.run(
        parameters={"ModelRunID": deploy_model_setup["run_id"]}
    )

    assert flow_run.is_successful()
