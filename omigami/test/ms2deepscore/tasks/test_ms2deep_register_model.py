import os
from pathlib import Path

import mlflow
import pytest
from mlflow.pyfunc import PyFuncModel

from omigami.ms2deepscore.predictor import MS2DeepScorePredictor

from omigami.ms2deepscore.tasks import (
    ModelRegister,
    RegisterModel,
    RegisterModelParameters,
)
from prefect import Flow

from omigami.ms2deepscore.helper_classes.siamese_model_trainer import SplitRatio
from omigami.ms2deepscore.tasks.train_model import TrainModelParameters

os.chdir(Path(__file__).parents[4])


@pytest.fixture
def train_parameters(ms2deepscore_model_path):

    return TrainModelParameters(
        ms2deepscore_model_path,
        "Test/Path/SpectrumBinner",
        500,
        0.3,
        30,
        60,
        0.5,
        SplitRatio(0.8, 0.1, 0.1),
    )


def test_register_model(ms2deepscore_model_path, tmpdir, train_parameters):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(path)

    _ = model_register.register_model(
        model=MS2DeepScorePredictor(ion_mode="positive"),
        experiment_name="experiment",
        output_path=path,
        train_parameters=train_parameters,
        artifacts={"ms2deepscore_model_path": ms2deepscore_model_path},
    )

    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")
    assert "ms2deepscore" in os.listdir(f"{path}/model/code/omigami")
    assert "ms2deepscore_model.hdf5" in os.listdir(f"{path}/model/artifacts")


def test_load_registered_model(ms2deepscore_model_path, tmpdir, train_parameters):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(path)

    model_register.register_model(
        model=MS2DeepScorePredictor(ion_mode="positive"),
        experiment_name="experiment",
        output_path=path,
        train_parameters=train_parameters,
        artifacts={"ms2deepscore_model_path": ms2deepscore_model_path},
    )

    loaded_model = mlflow.pyfunc.load_model(f"{path}/model")
    assert isinstance(loaded_model, PyFuncModel)


def test_model_register_task(ms2deepscore_model_path, tmpdir, train_parameters):
    path = f"{tmpdir}/mlflow/"
    parameters = RegisterModelParameters(
        experiment_name="experiment",
        mlflow_output_path=path,
        server_uri=path,
        ion_mode="positive",
    )
    with Flow("test") as flow:
        res = RegisterModel(parameters, train_parameters)(
            model_path=ms2deepscore_model_path
        )

    state = flow.run()
    assert state.is_successful()
    assert isinstance(state.result[res].result, dict)
    assert list(state.result[res].result.keys()) == ["model_uri", "run_id"]
