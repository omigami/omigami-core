import mlflow
import pytest
import numpy as np

from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.model_register import ModelRegister
from spec2vec_mlops.helper_classes.spec2vec_model import Model


@pytest.fixture
def saved_model_run_id(word2vec_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(f"file:/{path}")
    run_id = model_register.register_model(
        Model(
            word2vec_model,
            n_decimals=1,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5.0,
        ),
        "experiment",
        path,
    )
    return run_id


def test_pre_process_data(loaded_data, word2vec_model, documents_data):
    model = Model(
        word2vec_model,
        n_decimals=1,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
    )
    embeddings_from_model = model._pre_process_data([loaded_data[0]])

    em = EmbeddingMaker()
    embedding_from_flow = em.make_embedding(
        model=word2vec_model,
        document=documents_data[0],
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5.0,
    )
    assert all(embedding_from_flow == embeddings_from_model[0])


def test_predict_from_saved_model(saved_model_run_id, loaded_data):
    run = mlflow.get_run(saved_model_run_id)
    modelpath = f"{run.info.artifact_uri}/model/"
    model = mlflow.pyfunc.load_model(modelpath)
    embeddings = model.predict(loaded_data)
    assert isinstance(embeddings[0], np.ndarray)
