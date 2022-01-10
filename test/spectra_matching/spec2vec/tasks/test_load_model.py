import pytest
from gensim.models import Word2Vec

from omigami.spectra_matching.spec2vec.config import PREDICTOR_ENV_PATH
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
        Word2Vec(),
        ion_mode="positive",
        n_decimals=2,
        intensity_weighting_power=1.0,
        allowed_missing_percentage=15,
    )
    run_id = dgw.register_model(
        model=model,
        run_name="run",
        conda_env_path=PREDICTOR_ENV_PATH,
        experiment_name="test-experiment",
        model_name="test",
        experiment_path=str(tmpdir),
    )
    return {"uri": mlflow_uri, "run": run_id}


def test_load_spec2vec_model(mlflow_setup):
    task = LoadSpec2VecModel(FSDataGateway(), mlflow_setup["uri"])

    model = task.run(mlflow_setup["run"])

    assert isinstance(model, Word2Vec)
