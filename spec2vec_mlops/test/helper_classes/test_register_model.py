import os
import shutil

import mlflow
import pytest
from spec2vec_mlops.helper_classes.model_register import Model, ModelRegister
from spec2vec_mlops.helper_classes.model_trainer import ModelTrainer


@pytest.fixture
def trained_model(documents_data):
    model_trainer = ModelTrainer()
    return model_trainer.train_model(documents_data, iterations=10, window=5)


def test_register_model(trained_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    mlflow.set_registry_uri(f"{path}/mlflow")
    model_register = ModelRegister()
    model_register.register_model(Model(trained_model), f"{path}/model", n_decimals=2)
    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists("mlruns/0/")
    shutil.rmtree("mlruns")
