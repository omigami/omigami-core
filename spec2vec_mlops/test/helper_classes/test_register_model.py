import os
import pytest

from spec2vec_mlops.helper_classes.model_register import Model, ModelRegister
from spec2vec_mlops.helper_classes.model_trainer import ModelTrainer


@pytest.fixture
def trained_model(documents_data):
    model_trainer = ModelTrainer()
    return model_trainer.train_model(documents_data, iterations=10, window=5)


def test_get_or_create_experiment(tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(f"file:/{path}")
    experiment_name = "experiment"
    # create
    created_experiment_id = model_register._get_or_create_experiment_id(
        experiment_name, path
    )
    # get
    experiment_id = model_register._get_or_create_experiment_id(experiment_name, path)
    assert created_experiment_id == experiment_id


def test_register_model(trained_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(f"file:/{path}")
    model_register.register_model(
        Model(trained_model), "experiment", path, n_decimals=2
    )
    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")
