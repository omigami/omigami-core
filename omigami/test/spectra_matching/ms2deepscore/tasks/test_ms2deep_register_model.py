import os
from pathlib import Path

import mlflow
import pytest
from mlflow.pyfunc import PyFuncModel
from prefect import Flow

from omigami.config import MLFLOW_SERVER
from omigami.spectra_matching.ms2deepscore.helper_classes.siamese_model_trainer import (
    SplitRatio,
)
from omigami.spectra_matching.ms2deepscore.predictor import MS2DeepScorePredictor
from omigami.spectra_matching.ms2deepscore.tasks import (
    ModelRegister,
    RegisterModel,
    RegisterModelParameters,
    TrainModelParameters,
)

os.chdir(Path(__file__).parents[4])


@pytest.fixture
def train_parameters(ms2deepscore_model_path):
    return TrainModelParameters(
        output_path=ms2deepscore_model_path,
        ion_mode="positive",
        spectrum_binner_output_path="500",
        epochs=50,
        split_ratio=SplitRatio(0.8, 0.1, 0.1),
    )


def test_register_model(ms2deepscore_model_path, tmpdir, train_parameters):
    model_register = ModelRegister(MLFLOW_SERVER)

    run_id = model_register.register_model(
        model=MS2DeepScorePredictor("positive"),
        experiment_name="experiment",
        output_path=str(tmpdir),
        train_parameters=train_parameters,
        artifacts={"ms2deepscore_model_path": ms2deepscore_model_path},
    )

    output_dir = mlflow.get_run(run_id).info.artifact_uri
    assert set(os.listdir(f"{output_dir}/model")) == {
        "artifacts",
        "MLmodel",
        "code",
        "python_model.pkl",
        "conda.yaml",
    }
    assert (os.listdir(f"{output_dir}/model/artifacts")) == ["ms2deepscore_model.hdf5"]


def test_load_registered_model(ms2deepscore_model_path, tmpdir, train_parameters):
    model_register = ModelRegister(MLFLOW_SERVER)

    run_id = model_register.register_model(
        model=MS2DeepScorePredictor("positive"),
        experiment_name="experiment",
        output_path=str(tmpdir),
        train_parameters=train_parameters,
        artifacts={"ms2deepscore_model_path": ms2deepscore_model_path},
    )
    output_dir = mlflow.get_run(run_id).info.artifact_uri

    loaded_model = mlflow.pyfunc.load_model(f"{output_dir}/model")
    assert isinstance(loaded_model, PyFuncModel)


def test_model_register_task(
    ms2deepscore_model_path, tmpdir, train_parameters, mock_default_config
):
    parameters = RegisterModelParameters(
        experiment_name="experiment",
        mlflow_output_path=tmpdir,
        ion_mode="positive",
    )

    train_model_output = {
        "ms2deepscore_model_path": ms2deepscore_model_path,
        "validation_loss": 50,
    }

    with Flow("test") as flow:
        res = RegisterModel(parameters, train_parameters)(
            train_model_output=train_model_output
        )

    state = flow.run()
    assert state.is_successful()
    assert isinstance(state.result[res].result, dict)
    assert list(state.result[res].result.keys()) == ["model_uri", "run_id"]


def test_convert_train_parameters(ms2deepscore_model_path, tmpdir, train_parameters):
    model_register = ModelRegister(MLFLOW_SERVER)
    parameters = model_register._convert_train_parameters(
        train_parameters=train_parameters, validation_loss=0.4
    )

    expected_keys = [
        "epochs",
        "learning_rate",
        "layer_base_dims",
        "embedding_dim",
        "dropout_rate",
        "split_ratio",
        "validation_loss",
    ]

    assert len(parameters) == 7
    assert type(dict()) == type(parameters)
    assert list(parameters.keys()) == expected_keys
