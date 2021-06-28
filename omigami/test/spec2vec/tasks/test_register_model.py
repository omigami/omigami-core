import os
from pathlib import Path

from omigami.spec2vec.predictor import Spec2VecPredictor
from omigami.spec2vec.tasks.register_model import ModelRegister

os.chdir(Path(__file__).parents[4])


def test_register_model(word2vec_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(path)
    run_id = model_register.register_model(
        model=Spec2VecPredictor(
            word2vec_model,
            n_decimals=2,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5.0,
        ),
        experiment_name="experiment",
        output_path=path,
    )
    assert run_id
    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")
    assert "spec2vec" in os.listdir(f"{path}/model/code")
