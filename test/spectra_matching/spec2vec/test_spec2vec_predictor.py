import os
from pathlib import Path
from unittest.mock import Mock

import mlflow
import pandas as pd
import pytest
from pytest_redis import factories
from seldon_core.metrics import SeldonMetrics
from seldon_core.wrapper import get_rest_microservice

from omigami.spectra_matching.predictor import SpectraMatchingError
from omigami.spectra_matching.spec2vec.entities.embedding import Spec2VecEmbedding
from omigami.spectra_matching.spec2vec.helper_classes.embedding_maker import (
    EmbeddingMaker,
)
from omigami.spectra_matching.spec2vec.predictor import Spec2VecPredictor
from test.spectra_matching.conftest import MLFlowServer

redis_db = factories.redisdb("redis_nooproc")

os.chdir(Path(__file__).parents[3])


@pytest.fixture
def small_payload():
    small_payload = {
        "parameters": {"n_best_spectra": 2},
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
        "parameters": {"n_best_spectra": 2},
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
def spec2vec_predictor(word2vec_model):
    return Spec2VecPredictor(
        ion_mode="positive",
        n_decimals=1,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=25,
        run_id="1",
        model=word2vec_model,
    )


@pytest.fixture()
def predict_parameters():
    return {"n_best_spectra": 5}


def test_pre_process_data(
    word2vec_model, raw_spectra, spec2vec_predictor, documents_data
):
    data = [d for d in raw_spectra if d["SpectrumID"] == "CCMSLIB00000072099"]
    embeddings_from_model = spec2vec_predictor._pre_process_data(data)

    em = EmbeddingMaker(n_decimals=1)
    embedding_from_flow = em.make_embedding(
        model=word2vec_model,
        document=documents_data[0],
        intensity_weighting_power=0.5,
        allowed_missing_percentage=25.0,
    )
    assert all(embedding_from_flow.vector == embeddings_from_model[0].vector)


def test_get_best_matches(spec2vec_predictor, spec2vec_embeddings):
    n_best_spectra = 2
    best_matches = {}
    for query in spec2vec_embeddings[:2]:
        input_best_matches = spec2vec_predictor._calculate_best_matches(
            spec2vec_embeddings,
            query,
            n_best_spectra=n_best_spectra,
        )
        best_matches[query.spectrum_id] = input_best_matches

    for query, (best_match_id, best_match) in zip(
        spec2vec_embeddings, best_matches.items()
    ):
        assert len(best_match) == n_best_spectra
        assert query.spectrum_id == best_match_id
        assert "score" in pd.DataFrame(best_match).T.columns


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_local_predictions(big_payload, spec2vec_redis_setup, spec2vec_predictor):
    spec2vec_predictor._run_id = "1"

    matches_big = spec2vec_predictor.predict(
        data_input_and_parameters=big_payload, mz_range=10, context=""
    )

    assert len(matches_big) == 2
    assert len(matches_big["spectrum-0"]) == 2


def test_parse_input(small_payload, spec2vec_predictor):
    data_input, parameters = spec2vec_predictor._parse_input(small_payload)

    assert parameters["n_best_spectra"] == small_payload["parameters"]["n_best_spectra"]
    assert "peaks_json" and "Precursor_MZ" in data_input[0]


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_get_ref_ids_from_data_input(
    small_payload, spec2vec_predictor, spec2vec_redis_setup
):
    data_input = small_payload.get("data")
    spectrum_ids = spec2vec_predictor._get_ref_ids_from_data_input(data_input)

    assert spectrum_ids
    assert isinstance(spectrum_ids[0][0], str)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_load_unique_ref_embeddings(spec2vec_predictor, spec2vec_redis_setup):
    spectrum_ids = [["CCMSLIB00000006878", "CCMSLIB00000007092"]]

    ref_embeddings = spec2vec_predictor._load_unique_ref_embeddings(spectrum_ids)

    assert ref_embeddings
    assert isinstance(ref_embeddings["CCMSLIB00000006878"], Spec2VecEmbedding)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_get_input_ref_embeddings(spec2vec_predictor, spec2vec_redis_setup):
    input_ref_spectrum_ids = ["CCMSLIB00000006878", "CCMSLIB00000007092"]
    all_ref_spectrum_ids = [
        [
            "CCMSLIB00000006878",
            "CCMSLIB00000007092",
            "CCMSLIB00000070262",
            "CCMSLIB00000070272",
        ]
    ]
    ref_embeddings = spec2vec_predictor._load_unique_ref_embeddings(
        all_ref_spectrum_ids
    )

    ref_emb_for_input = spec2vec_predictor._get_input_ref_embeddings(
        input_ref_spectrum_ids, ref_embeddings
    )

    assert isinstance(ref_emb_for_input[0], Spec2VecEmbedding)
    assert len(ref_emb_for_input) == len(input_ref_spectrum_ids)
    assert ref_emb_for_input[0].spectrum_id == input_ref_spectrum_ids[0]


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_add_metadata(spec2vec_predictor, spec2vec_embeddings, spec2vec_redis_setup):
    n_best_spectra = 3
    best_matches = {}
    for i, query in enumerate(spec2vec_embeddings):
        input_best_matches = spec2vec_predictor._calculate_best_matches(
            spec2vec_embeddings,
            query,
            n_best_spectra=n_best_spectra,
        )
        best_matches[query.spectrum_id] = input_best_matches

    best_matches = spec2vec_predictor._add_metadata(best_matches)

    assert len(best_matches) == 100
    bm = best_matches["CCMSLIB00000072099"]
    assert set(bm["CCMSLIB00000072099"].keys()) == {"score", "metadata"}
    assert len(bm["CCMSLIB00000072099"]["metadata"]) == 37  # nr of metadata in GNPS
    assert bm["CCMSLIB00000072099"]["metadata"]["compound_name"] == "Coproporphyrin I"


def test_predictor_error_handling(tmpdir):
    predictor = Spec2VecPredictor(
        ion_mode="positive",
        n_decimals=1,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=25,
        run_id="1",
    )

    predictor.predict = Mock(side_effect=SpectraMatchingError("error", 1, 400))

    metrics = Mock(spec=SeldonMetrics)
    app = get_rest_microservice(predictor, metrics)
    client = app.test_client()

    response = client.get('/predict?json={"data":{"ndarray":[1,2]}}')

    assert response.status_code == 400
    assert response.json["status"]["app_code"] == 1


@pytest.mark.xfail
def test_predictor_error_handling_mlflow(registered_s2v_model):
    run = mlflow.get_run(registered_s2v_model["run_id"])
    model_uri = run.info.artifact_uri + "/model"
    predictor = mlflow.pyfunc.load_model(model_uri)

    predictor.predict = Mock(side_effect=SpectraMatchingError("error", 1, 400))

    metrics = Mock(spec=SeldonMetrics)
    app = get_rest_microservice(predictor, metrics)
    client = app.test_client()

    response = client.get('/predict?json={"data":{"ndarray":[1,2]}}')

    assert response.status_code == 400
    assert response.json["status"]["app_code"] == 1


def test_seldon_internal_error_message(registered_s2v_model):
    run = mlflow.get_run(registered_s2v_model["run_id"])
    model_uri = run.info.artifact_uri + "/model"
    predictor = mlflow.pyfunc.load_model(model_uri)

    predictor._model_impl.python_model._get_ref_ids_from_data_input = Mock(
        side_effect=RuntimeError("Funtime error.")
    )

    payload = {
        "data": {
            "ndarray": {
                "parameters": {
                    "n_best_spectra": 2,
                },
                "data": [
                    {
                        "peaks_json": "[[289.286377,8068.000000],[295.545288,22507.000000]]",
                        "Precursor_MZ": "900",
                    },
                    {
                        "peaks_json": "[[289.286377,8068.000000],[295.545288,22507.000000]]",
                        "Precursor_MZ": "800",
                    },
                ],
            }
        }
    }

    metrics = Mock(spec=SeldonMetrics)
    app = get_rest_microservice(MLFlowServer(predictor), metrics)
    client = app.test_client()

    response = client.post("/predict", json=payload)

    assert response.status_code == 500
    assert (
        response.json["status"]["info"] == "SpectraMatchingError 500: 'Funtime error.'"
    )
