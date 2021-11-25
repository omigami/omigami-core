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
            "positive",
            n_decimals=1,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5.0,
        ),
        experiment_name="experiment",
        output_path=path,
    )
    assert run_id
    model_files = os.listdir(f"{path}/{run_id}/artifacts/model")
    assert set(model_files) == {"MLmodel", "code", "python_model.pkl", "conda.yaml"}
    assert "spec2vec" in os.listdir(f"{path}/{run_id}/artifacts/model/code/omigami")
