import os

from spec2vec_mlops.helper_classes.model_register import Model, ModelRegister


def test_register_model(word2vec_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(f"file:/{path}")
    model_register.register_model(
        Model(word2vec_model), "experiment", f"{path}/model", n_decimals=2
    )
    assert os.path.exists(f"{path}/model/python_model.pkl")


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


# model_register.register_model can not be tested locally. It throws the following error at mlflow.pyfunc.log_model:
# mlflow.exceptions.MlflowException: Model Registry features are not supported by the store with URI: {tmpdir}.
# Stores with the following URI schemes are supported:
# ['databricks', 'http', 'https', 'postgresql', 'mysql', 'sqlite', 'mssql'].
# This means we would need to use the MLFlow local host and setup a database to test the mlflow.pyfunc.log_model
