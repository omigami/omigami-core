import pytest

from spec2vec_mlops.helper_classes.data_loader import DataLoader
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.model_register import ModelRegister
from spec2vec_mlops.helper_classes.spec2vec_model import Model


@pytest.fixture
def saved_model(word2vec_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(f"file:/{path}")
    run_id = model_register.register_model(
        Model(
            word2vec_model,
            n_decimals=2,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5.0,
        ),
        "experiment",
        path,
    )
    return run_id


def test_pre_process_data(gnps_small_json, word2vec_model, documents_data):
    dl = DataLoader()
    loaded_data = dl.load_gnps_json(gnps_small_json)
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


def test_predict_from_saved_model(saved_model):
    pass
