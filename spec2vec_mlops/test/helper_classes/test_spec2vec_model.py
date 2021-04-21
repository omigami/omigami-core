import json
import os
from pathlib import Path

import mlflow
import pytest
from seldon_core.metrics import SeldonMetrics
from seldon_core.wrapper import get_rest_microservice

from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.exception import (
    MandatoryKeyMissingException,
    IncorrectSpectrumDataTypeException,
    IncorrectPeaksJsonTypeException,
    IncorrectFloatFieldTypeException,
    IncorrectStringFieldTypeException,
)
from spec2vec_mlops.helper_classes.model_register import ModelRegister
from spec2vec_mlops.helper_classes.spec2vec_model import Model

os.chdir(Path(__file__).parents[3])


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


@pytest.fixture()
def model(word2vec_model):
    return Model(
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
        assert query.spectrum_id == best_match[0]["best_match_id"]


@pytest.mark.skipif(
    os.getenv("SKIP_SPARK_TEST", True),
    reason="It can only be run if the Feast docker-compose is up and with Spark",
)
def test_get_reference_embeddings(model, all_spectrum_ids, embeddings_stored):
    embeddings = model._get_reference_embeddings()
    assert len(embeddings) == len(all_spectrum_ids)


@pytest.mark.skipif(
    os.getenv("SKIP_SPARK_TEST", True),
    reason="It can only be run if the Feast docker-compose is up and with Spark",
)
def test_predict_from_saved_model(
    saved_model_run_id, loaded_data, embeddings_stored, predict_parameters
):
    run = mlflow.get_run(saved_model_run_id)
    model_path = f"{run.info.artifact_uri}/model/"
    model = mlflow.pyfunc.load_model(model_path)
    data_and_param = loaded_data + [predict_parameters]
    best_matches = model.predict(data_and_param)
    for spectrum, best_match in zip(loaded_data, best_matches):
        assert len(best_match) == predict_parameters["n_best_spectra"]


def test_raise_api_exception(model):
    user_object = Model(
        model, n_decimals=1, intensity_weighting_power=0.5, allowed_missing_percentage=5
    )
    seldon_metrics = SeldonMetrics()
    app = get_rest_microservice(user_object, seldon_metrics)
    client = app.test_client()
    rv = client.get('/predict?json={"data":{"ndarray":[[1,2]], "names":["spec1"]}}')
    j = json.loads(rv.data)
    assert rv.status_code == 400
    assert j["status"]["info"] == "Spectrum data must be a dictionary"
