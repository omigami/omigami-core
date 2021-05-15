import json
import os
import pickle
from pathlib import Path
from typing import List

import mlflow
import pytest
from pytest_redis import factories
from seldon_core.metrics import SeldonMetrics
from seldon_core.wrapper import get_rest_microservice

from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.exception import (
    MandatoryKeyMissingException,
    IncorrectSpectrumDataTypeException,
    IncorrectPeaksJsonTypeException,
    IncorrectFloatFieldTypeException,
    IncorrectStringFieldTypeException,
)
from spec2vec_mlops.tasks.register_model import ModelRegister
from spec2vec_mlops.predictor import Predictor
from spec2vec_mlops.test.conftest import ASSETS_DIR

EMBEDDING_HASHES = config["redis"]["embedding_hashes"]

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
            },
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
                "[131.083939, 917.0], [133.101639, 75.0], [135.11908, 203.0], [136.925232, 163.0], "
                "[141.068222, 331.0], [142.082947, 193.0], [143.088394, 623.0], [143.576874, 70.0], "
                "[145.07016, 170.0], [145.105026, 379.0], [147.124954, 63.0], [149.134857, 156.0], "
                "[154.542526, 94.0], [159.119614, 289.0], [161.135452, 111.0], [164.157211, 433.0], "
                "[165.069885, 55.0], [166.07579, 163.0], [171.12529, 261.0], [173.09967, 182.0], "
                "[185.066528, 149.0], [185.135742, 230.0], [189.129272, 164.0], [193.12529, 121.0], "
                "[196.12677, 166.0], [199.114395, 225.0], [199.154922, 102.0], [207.142349, 179.0], "
                "[212.152267, 211.0], [213.134125, 397.0], [214.168991, 100.0], [217.117264, 145.0], "
                "[217.153534, 170.0], [229.207336, 199.0], [229.305023, 106.0], [255.176453, 162.0], "
                "[259.16922, 175.0], [273.219421, 190.0], [301.217468, 592.0], [1337.508301, 230.0]]",
                "Precursor_MZ": "301.216",
            }
        ],
    }
    return big_payload


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


@pytest.mark.parametrize(
    "input_data, exception",
    [
        ([[]], IncorrectSpectrumDataTypeException),
        ([{}], MandatoryKeyMissingException),
        ([{"peaks_json": ""}], MandatoryKeyMissingException),
        (
            [{"peaks_json": "peaks", "Precursor_MZ": "1.0"}],
            IncorrectPeaksJsonTypeException,
        ),
        (
            [{"peaks_json": {}, "Precursor_MZ": "1.0"}],
            IncorrectPeaksJsonTypeException,
        ),
        (
            [{"peaks_json": [], "Precursor_MZ": "some_mz"}],
            IncorrectFloatFieldTypeException,
        ),
        (
            [{"peaks_json": [], "Precursor_MZ": "1.0", "INCHI": 1}],
            IncorrectStringFieldTypeException,
        ),
    ],
)
def test_validate_input_raised_expections(input_data, exception, model):
    with pytest.raises(exception):
        model._validate_input(input_data)


def test_validate_input_valid(model):
    model._validate_input(
        [{"peaks_json": [], "Precursor_MZ": "1.0", "INCHI": "some_key"}],
    )


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
    best_matches = model._get_best_matches(embeddings, embeddings, n_best_spectra)
    for query, best_match in zip(embeddings, best_matches):
        assert len(best_match) == n_best_spectra
        assert query.spectrum_id == best_match[0]["match_spectrum_id"]


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_get_reference_embeddings(model, embeddings, redis_db):
    run_id = "1"
    pipe = redis_db.pipeline()
    for embedding in embeddings:
        pipe.hmset(
            f"{EMBEDDING_HASHES}_{run_id}",
            {embedding.spectrum_id: pickle.dumps(embedding)},
        )
    pipe.execute()

    embeddings_read = model._get_reference_embeddings()
    assert len(embeddings) == len(embeddings_read)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_predict_from_saved_model(
    saved_model_run_id, loaded_data, predict_parameters, embeddings
):
    pipe = redis_db.pipeline()
    for embedding in embeddings:
        pipe.hset(
            f"{EMBEDDING_HASHES}_{saved_model_run_id}",
            embedding.spectrum_id,
            pickle.dumps(embedding),
        )
    pipe.execute()

    run = mlflow.get_run(saved_model_run_id)
    model_path = f"{run.info.artifact_uri}/model/"
    model = mlflow.pyfunc.load_model(model_path)
    data_and_param = {"parameters": predict_parameters, "data": loaded_data}
    best_matches = model.predict(data_and_param)
    for spectrum, best_match in zip(loaded_data, best_matches):
        assert len(best_match) == predict_parameters["n_best_spectra"]
        assert best_match[0]["match_spectrum_id"] == spectrum["spectrum_id"]


def test_local_predictions(small_payload, big_payload):
    path = str(ASSETS_DIR / "full_data/new_predictor.pkl")
    with open(path, "rb") as input_file:
        local_model = pickle.load(input_file)

    matches_small = local_model.predict(
        model_input_and_parameters=small_payload, context=""
    )
    matches_big = local_model.predict(
        model_input_and_parameters=big_payload, context=""
    )

    assert isinstance(matches_small, List)
    assert isinstance(matches_big, List)


def test_10k_predictor():
    model_uri = "s3://dr-prefect/spec2vec-training-flow/mlflow/tests/e06d4ef7116e4bc78b76fc867fff29dc/artifacts/model"
    predictor = Predictor(
        model, n_decimals=1, intensity_weighting_power=0.5, allowed_missing_percentage=5
    )
