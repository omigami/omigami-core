import os
from copy import deepcopy
from pathlib import Path

import mlflow
import pytest

from omigami.ms2deepscore.predictor import Predictor
from omigami.ms2deepscore.tasks.register_model import ModelRegister

os.chdir(Path(__file__).parents[4])


@pytest.mark.skip("Need to fix how to serialize the MS2DeepScor")
def test_register_model(ms2deepscore_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(path)
    run_id = model_register.register_model(
        Predictor(ms2deepscore_model),
        "experiment",
        path,
    )
    assert run_id
    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")
    assert "ms2deepscore" in os.listdir(f"{path}/model/code")


# def test_load_registered_model(ms2deepscore_model, tmpdir):
#     path = f"{tmpdir}/mlflow/"
#     model_register = ModelRegister(path)
#     model_register.register_model(
#         Predictor(ms2deepscore_model),
#         "experiment",
#         path,
#     )
#
#     model = mlflow.pyfunc.load_model(path)
#     assert isinstance(model, Predictor)
