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
    IncorrectSpectrumNameTypeException,
    IncorrectDataLengthException,
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
    )


@pytest.mark.parametrize(
    "context, input_data, exception",
    [
        (
            ["1", "2"],
            [{"peaks_json": []}],
            IncorrectDataLengthException,
        ),
        (
            [1],
            [{"peaks_json": []}],
            IncorrectSpectrumNameTypeException,
        ),
        (["spectrum"], [[]], IncorrectSpectrumDataTypeException),
        (["spectrum"], [{}], MandatoryKeyMissingException),
        (
            ["spectrum"],
            [{"peaks_json": "peaks"}],
            IncorrectPeaksJsonTypeException,
        ),
        (
            ["spectrum"],
            [{"peaks_json": {}}],
            IncorrectPeaksJsonTypeException,
        ),
        (
            None,
            [{"peaks_json": [], "Precursor_MZ": "some_mz"}],
            IncorrectFloatFieldTypeException,
        ),
        (
            None,
            [{"peaks_json": [], "Precursor_MZ": "1.0", "INCHI": 1}],
            IncorrectStringFieldTypeException,
        ),
    ],
)
def test_validate_input_raised_expections(context, input_data, exception, model):
    with pytest.raises(exception):
        model._validate_input(context, input_data)


def test_validate_input_valid(model):
    model._validate_input(
        ["spectrum_name"],
        [{"peaks_json": [], "INCHI": "some_key"}],
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
    best_matches = model._get_best_matches(
        embeddings[:50], embeddings[50:], ["spectrum_name"] * 50
    )
    assert all(
        key in best_matches[0] for key in ["spectrum_name", "best_match_id", "score"]
    )


def test_predict_from_saved_model(saved_model_run_id, loaded_data):
    run = mlflow.get_run(saved_model_run_id)
    modelpath = f"{run.info.artifact_uri}/model/"
    model = mlflow.pyfunc.load_model(modelpath)
    best_matches = model.predict(loaded_data)
    for spectrum in best_matches:
        assert spectrum["best_match_id"] is not None


def test_raise_api_exception(model):
    user_object = Model(
        model, n_decimals=1, intensity_weighting_power=0.5, allowed_missing_percentage=5
    )
    seldon_metrics = SeldonMetrics()
    app = get_rest_microservice(user_object, seldon_metrics)
    client = app.test_client()
    rv = client.get(
        '/predict?json={"data":{"names":["spectrum1","spectrum2"],"ndarray":[[1,2]]}}'
    )
    j = json.loads(rv.data)
    assert rv.status_code == 400
    assert j["status"]["info"] == "Spectrum data must be a dictionary"
