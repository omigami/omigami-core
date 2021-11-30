from unittest.mock import Mock

import pytest

from omigami.spectra_matching.spec2vec.predictor import Spec2VecPredictor
from omigami.spectra_matching.spec2vec.tasks.deploy_model_tasks import LoadSpec2VecModel
from omigami.spectra_matching.storage import FSDataGateway
from omigami.spectra_matching.storage.model_registry import MLFlowDataGateway


@pytest.fixture(scope="module")
def mlflow_setup(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("model")
    mlflow_uri = f"sqlite:///{tmpdir}/mlflow.sqlite"
    dgw = MLFlowDataGateway(mlflow_uri)
    model = Spec2VecPredictor(
        Mock(),
        ion_mode="positive",
        n_decimals=2,
        intensity_weighting_power=1.0,
        allowed_missing_percentage=15,
    )
    run_id = dgw.register_model(
        model=model,
        run_name="run",
        experiment_name="test-experiment",
        model_name="test",
        experiment_path=str(tmpdir),
    )
    return {"model": model, "path": tmpdir, "dgw": dgw, "run": run_id}


def test_load_spec2vec_model(mlflow_setup):
    fs_dgw = FSDataGateway()
    mlflow_dgw = mlflow_setup["dgw"]
    task = LoadSpec2VecModel(mlflow_setup["run"], fs_dgw, mlflow_dgw)

    model = task.run()

    assert isinstance(model, Spec2VecPredictor)
