from pathlib import Path

from omigami.config import MLFLOW_DIRECTORY
from omigami.spectra_matching.ms2deepscore.tasks import (
    GetMS2DeepScoreModelPath,
    RegisterModel,
    RegisterModelParameters,
    TrainModelParameters,
)


def test_get_ms2ds_model_path(siamese_model_path, tmpdir):
    mlflow_uri = f"sqlite:///{tmpdir}/mlflow.sqlite"
    params = RegisterModelParameters(
        "test_experiment", mlflow_uri, MLFLOW_DIRECTORY, "positive"
    )
    train_params = TrainModelParameters("path", "positive", "path")
    register_task = RegisterModel(params, train_params)
    model_run_id = register_task.run({"ms2deepscore_model_path": siamese_model_path})

    get_model_path_task = GetMS2DeepScoreModelPath(mlflow_uri)

    path = get_model_path_task.run(model_run_id)

    assert Path(path["model_path"]).exists()
