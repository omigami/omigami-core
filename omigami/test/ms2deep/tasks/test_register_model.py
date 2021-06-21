import os
from pathlib import Path

from omigami.ms2deep.predictor import Predictor
from omigami.ms2deep.tasks.register_model import ModelRegister

os.chdir(Path(__file__).parents[4])


def test_register_model(word2vec_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(path)
    run_id = model_register.register_model(
        Predictor(),
        "experiment",
        path,
    )
    assert run_id
    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")
    assert "ms2deep" in os.listdir(f"{path}/model/code")
