import os
from pathlib import Path

from omigami.spec2vec.tasks.register_model import ModelRegister
from omigami.spec2vec.predictor import Predictor

os.chdir(Path(__file__).parents[4])


def test_get_or_create_experiment(tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(path)
    experiment_name = "experiment"
    # create
    created_experiment_id = model_register._get_or_create_experiment_id(
        experiment_name, path
    )
    # get
    experiment_id = model_register._get_or_create_experiment_id(experiment_name, path)
    assert created_experiment_id == experiment_id


def test_register_model(word2vec_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(path)
    run_id = model_register.register_model(
        Predictor(
            word2vec_model,
            n_decimals=2,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5.0,
        ),
        "experiment",
        path,
    )
    assert run_id
    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")
