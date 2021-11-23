import os
from pathlib import Path

import mlflow
import pytest
from mlflow.pyfunc import PythonModel

from omigami.model_register import MLFlowModelRegister
from omigami.spec2vec.predictor import Spec2VecPredictor

os.chdir(Path(__file__).parents[2])


class MockABCModelRegister(MLFlowModelRegister):
    def register_model(
        self,
        model: PythonModel,
        experiment_name: str,
        path: str,
        conda_env_path: str = None,
    ):
        pass


def test_get_or_create_experiment(tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = MockABCModelRegister(path)
    experiment_name = "experiment"
    # create
    created_experiment_id = model_register._get_or_create_experiment_id(
        experiment_name, path
    )
    # get
    experiment_id = model_register._get_or_create_experiment_id(experiment_name, path)
    assert created_experiment_id == experiment_id


def test_save_model(word2vec_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = MockABCModelRegister(path)

    model_register.log_model(
        model=Spec2VecPredictor(
            word2vec_model,
            ion_mode="positive",
            n_decimals=1,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5.0,
        ),
        experiment_name="experiment",
        output_path=path,
        code_path=["omigami"],
    )

    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")


@pytest.mark.skip(
    "Requires MLFlow Server running locally."
    "mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root /tmp/mlflow"
)
def test_log_model(word2vec_model, tmpdir):

    model_register = MockABCModelRegister("http://localhost:5000")
    exp_id = model_register._get_or_create_experiment_id("experiment")

    with mlflow.start_run(experiment_id=exp_id, nested=True):
        model_register.log_model(
            model=Spec2VecPredictor(
                word2vec_model,
                ion_mode="positive",
                n_decimals=1,
                intensity_weighting_power=0.5,
                allowed_missing_percentage=5.0,
            ),
            experiment_name="experiment",
            code_path=["omigami"],
        )

        path = mlflow.get_artifact_uri()
        assert os.path.exists(f"{path}/model/python_model.pkl")
        assert os.path.exists(f"{path}/model/conda.yaml")
