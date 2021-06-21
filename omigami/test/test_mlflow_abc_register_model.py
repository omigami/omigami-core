import os
from pathlib import Path

from mlflow.pyfunc import PythonModel
from omigami.model_register import MLFlowModelRegister
from omigami.spec2vec.predictor import Predictor

os.chdir(Path(__file__).parents[2])


class TestABCModelRegister(MLFlowModelRegister):
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
    model_register = TestABCModelRegister(path)
    experiment_name = "experiment"
    # create
    created_experiment_id = model_register._get_or_create_experiment_id(
        experiment_name, path
    )
    # get
    experiment_id = model_register._get_or_create_experiment_id(experiment_name, path)
    assert created_experiment_id == experiment_id


def test_log_model(word2vec_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = TestABCModelRegister(path)

    model_register.log_model(
        model=Predictor(
            word2vec_model,
            n_decimals=2,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5.0,
        ),
        experiment_name="experiment",
        path=path,
        code_path=["omigami/spec2vec"],
    )

    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")
