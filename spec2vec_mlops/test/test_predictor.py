import os
import pickle
from pathlib import Path

import pytest
from pytest_redis import factories

from spec2vec_mlops.config import (
    EMBEDDING_HASHES,
    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
    SPECTRUM_HASHES,
)
from spec2vec_mlops.entities.embedding import Embedding
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker

from spec2vec_mlops.tasks.register_model import ModelRegister
from spec2vec_mlops.predictor import Predictor
from spec2vec_mlops.test.conftest import ASSETS_DIR


redis_db = factories.redisdb("redis_nooproc")

os.chdir(Path(__file__).parents[3])


@pytest.fixture
def small_payload():
    small_payload = {
        "parameters": {"n_best_spectra": 10},
        "data": [
            {
                "peaks_json": "[[80.060677, 157.0], [337.508301, 230.0]]",
                "Precursor_MZ": "353.233",
            }
        ],
    }
    return small_payload


@pytest.fixture
def big_payload():
    big_payload = {
        "parameters": {"n_best_spectra": 10},
        "data": [
            {
                "peaks_json": "[[80.060677, 157.0], [81.072548, 295.0], [83.088249, 185.0], [91.057564, 601.0], "
                "[95.085892, 634.0], [105.07077, 590.0], [107.09063, 228.0], [109.107414, 238.0], "
                "[115.056114, 482.0], [117.07032, 715.0], [120.779747, 313.0], [121.067802, 257.0], "
                "[121.102318, 430.0], [128.064728, 119.0], [129.074646, 241.0], [130.083542, 291.0], "
                "[259.16922, 175.0], [273.219421, 190.0], [301.217468, 592.0], [1337.508301, 230.0]]",
                "Precursor_MZ": "301.216",
            },
            {
                "peaks_json": "[[80.060677, 157.0], [81.072548, 295.0], [83.088249, 185.0], [91.057564, 601.0], "
                "[95.085892, 634.0], [105.07077, 590.0], [107.09063, 228.0], [109.107414, 238.0], "
                "[115.056114, 482.0], [117.07032, 715.0], [120.779747, 313.0], [121.067802, 257.0], "
                "[121.102318, 430.0], [128.064728, 119.0], [129.074646, 241.0], [130.083542, 291.0], "
                "[259.16922, 175.0], [273.219421, 190.0], [301.217468, 592.0], [1337.508301, 230.0]]",
                "Precursor_MZ": "320.90",
            },
        ],
    }
    return big_payload


@pytest.fixture()
def spectra_and_embeddings_stored(redis_db, cleaned_data, embeddings):
    run_id = "1"
    pipe = redis_db.pipeline()
    for embedding in embeddings:
        pipe.hset(
            f"{EMBEDDING_HASHES}_{run_id}",
            embedding.spectrum_id,
            pickle.dumps(embedding),
        )
    for spectrum in cleaned_data:
        pipe.zadd(
            SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
            {spectrum.metadata["spectrum_id"]: spectrum.metadata["precursor_mz"]},
        )
        pipe.hset(
            SPECTRUM_HASHES, spectrum.metadata["spectrum_id"], pickle.dumps(spectrum)
        )
    pipe.execute()


@pytest.fixture
def saved_model_run_id(word2vec_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(f"file:/{path}")
    run_id = model_register.register_model(
        Predictor(
            word2vec_model,
            n_decimals=1,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5.0,
        ),
        "experiment",
        path,
    )
    return run_id


@pytest.fixture()
def model(word2vec_model):
    return Predictor(
        word2vec_model,
        n_decimals=1,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        run_id="1",
    )


@pytest.fixture()
def predict_parameters():
    return {"n_best_spectra": 5}


def test_pre_process_data(word2vec_model, loaded_data, model, documents_data):
    embeddings_from_model = model._pre_process_data(loaded_data)

    em = EmbeddingMaker(n_decimals=1)
    embedding_from_flow = em.make_embedding(
        model=word2vec_model,
        document=documents_data[0],
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5.0,
    )
    assert all(embedding_from_flow.vector == embeddings_from_model[0].vector)


def test_get_best_matches(model, embeddings):
    n_best_spectra = 2
    best_matches = []
    for query in embeddings:
        input_best_matches = model._get_best_matches(
            embeddings,
            query,
            input_spectrum_number=0,
            n_best_spectra=n_best_spectra,
        )
        best_matches.append(input_best_matches)
    for query, best_match in zip(embeddings, best_matches):
        assert len(best_match) == n_best_spectra
        assert query.spectrum_id == best_match[0]["match_spectrum_id"]


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True) or os.getenv("CI", "False").title(),
    reason="It can only be run if the Redis is up",
)
# @pytest.mark.skipif(
#     not os.path.exists(str(ASSETS_DIR / "full_data/test_model.pkl")),
#     reason="test_model.pkl is git ignored",
# )
def test_local_predictions(small_payload, big_payload, spectra_and_embeddings_stored):
    path = str(ASSETS_DIR / "full_data/test_model.pkl")

    with open(path, "rb") as input_file:
        local_model = pickle.load(input_file)

    local_model.run_id = "1"

    matches_big = local_model.predict(
        data_input_and_parameters=big_payload, mz_range=10, context=""
    )
    matches_small = local_model.predict(
        data_input_and_parameters=small_payload, mz_range=10, context=""
    )

    assert len(matches_small[0]) != 0
    assert len(matches_big[0]) != 0


def test_parse_input(small_payload, model):
    data_input, parameters = model._parse_input(small_payload)

    assert parameters["n_best_spectra"] == 10
    assert "peaks_json" and "Precursor_MZ" in data_input[0]


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_get_ref_ids_from_data_input(
    small_payload, model, spectra_and_embeddings_stored
):
    data_input = small_payload.get("data")
    spectrum_ids = model._get_ref_ids_from_data_input(data_input)

    assert spectrum_ids
    assert isinstance(spectrum_ids[0][0], str)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_load_unique_ref_embeddings(model, spectra_and_embeddings_stored):
    spectrum_ids = [["CCMSLIB00000006878", "CCMSLIB00000007092"]]

    ref_embeddings = model._load_unique_ref_embeddings(spectrum_ids)

    assert ref_embeddings
    assert isinstance(ref_embeddings["CCMSLIB00000006878"], Embedding)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_get_input_ref_embeddings(model, spectra_and_embeddings_stored):
    input_ref_spectrum_ids = ["CCMSLIB00000006878", "CCMSLIB00000007092"]
    all_ref_spectrum_ids = [
        [
            "CCMSLIB00000006878",
            "CCMSLIB00000007092",
            "CCMSLIB00000070262",
            "CCMSLIB00000070272",
        ]
    ]
    ref_embeddings = model._load_unique_ref_embeddings(all_ref_spectrum_ids)

    ref_emb_for_input = model._get_input_ref_embeddings(
        input_ref_spectrum_ids, ref_embeddings
    )

    assert isinstance(ref_emb_for_input[0], Embedding)
    assert len(ref_emb_for_input) == len(input_ref_spectrum_ids)
    assert ref_emb_for_input[0].spectrum_id == input_ref_spectrum_ids[0]
