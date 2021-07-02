import os
from pathlib import Path

import mlflow
from mlflow.pyfunc import PyFuncModel
from prefect import Flow

from omigami.ms2deepscore.predictor import MS2DeepScorePredictor
from omigami.ms2deepscore.tasks.register_model import ModelRegister, RegisterModel

os.chdir(Path(__file__).parents[4])


def test_register_model(ms2deepscore_model_path, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(path)

    _ = model_register.register_model(
        model=MS2DeepScorePredictor(),
        experiment_name="experiment",
        output_path=path,
        artifacts={"ms2deepscore_model_path": ms2deepscore_model_path},
    )

    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")
    assert "ms2deepscore" in os.listdir(f"{path}/model/code/omigami")
    assert "ms2deepscore_model.hdf5" in os.listdir(f"{path}/model/artifacts")


def test_load_registered_model(ms2deepscore_model_path, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(path)

    model_register.register_model(
        model=MS2DeepScorePredictor(),
        experiment_name="experiment",
        output_path=path,
        artifacts={"ms2deepscore_model_path": ms2deepscore_model_path},
    )

    loaded_model = mlflow.pyfunc.load_model(f"{path}/model")
    assert isinstance(loaded_model, PyFuncModel)


def test_model_register_task(ms2deepscore_model_path, tmpdir):
    path = f"{tmpdir}/mlflow/"

    with Flow("test") as flow:
        res = RegisterModel(
            experiment_name="experiment", mlflow_output_path=path, server_uri=path
        )(model_path=ms2deepscore_model_path)

    state = flow.run()
    assert state.is_successful()
    assert isinstance(state.result[res].result, dict)
    assert list(state.result[res].result.keys()) == ["model_uri", "run_id"]
