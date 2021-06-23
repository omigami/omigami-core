import os
from pathlib import Path
import mlflow
from omigami.ms2deepscore.predictor import Predictor
from omigami.ms2deepscore.tasks.register_model import ModelRegister


os.chdir(Path(__file__).parents[4])


def test_register_model(ms2deepscore_model_path, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(path)

    run_id = model_register.register_model(
        model=Predictor(),
        experiment_name="experiment",
        path=path,
        artifacts={"ms2deepscore_model_path": ms2deepscore_model_path},
    )
    assert run_id
    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")
    assert "ms2deepscore" in os.listdir(f"{path}/model/code")
    assert "ms2deepscore_model.hdf5" in os.listdir(f"{path}/model/artifacts")


def test_load_registered_model(
    ms2deepscore_model_path, payload_identical_spectra, tmpdir
):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(path)

    model_register.register_model(
        model=Predictor(),
        experiment_name="experiment",
        path=path,
        artifacts={"ms2deepscore_model_path": ms2deepscore_model_path},
    )

    loaded_model = mlflow.pyfunc.load_model(f"{path}/model")
    assert loaded_model.predict(payload_identical_spectra) == 1
