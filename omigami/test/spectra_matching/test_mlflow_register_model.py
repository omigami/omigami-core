import os
from pathlib import Path

import mlflow
import pytest
from mlflow.pyfunc import PythonModel

from omigami.config import CODE_PATH
from omigami.spectra_matching.model_register import MLFlowModelRegister
from omigami.spectra_matching.spec2vec.predictor import Spec2VecPredictor

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


def test_save_model(tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = MockABCModelRegister(path)

    model_register.log_model(
        model=Spec2VecPredictor(
            "Model",
            ion_mode="positive",
            n_decimals=1,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5.0,
        ),
        model_name="experiment",
        output_path=path,
        code_path=[str(CODE_PATH)],
    )

    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")


@pytest.mark.skip(
    "Requires MLFlow Server running locally."
    "mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root /local-deployment/mlflow"
)
def test_log_model(tmpdir):

    model_register = MockABCModelRegister("http://localhost:5000")
    exp_id = model_register._get_or_create_experiment_id("experiment")

    class Word2Vec:
        def __init__(self):
            pass

    word2vec_model = Word2Vec()

    with mlflow.start_run(experiment_id=exp_id, nested=True):
        model_register.log_model(
            model=Spec2VecPredictor(
                word2vec_model,
                ion_mode="positive",
                n_decimals=1,
                intensity_weighting_power=0.5,
                allowed_missing_percentage=5.0,
            ),
            model_name="experiment",
            code_path=["omigami"],
        )

        path = mlflow.get_artifact_uri()
        assert os.path.exists(f"{path}/model/python_model.pkl")
        assert os.path.exists(f"{path}/model/conda.yaml")