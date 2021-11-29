import pickle

import pandas as pd
import pytest
from ms2deepscore.models import load_model
from pytest_redis import factories

import omigami.spectra_matching.ms2deepscore.helper_classes.siamese_model_trainer
from omigami.spectra_matching.ms2deepscore.config import BINNED_SPECTRUM_HASHES
from omigami.spectra_matching.ms2deepscore.helper_classes.spectrum_processor import (
    SpectrumProcessor,
)
from omigami.spectra_matching.ms2deepscore.predictor import MS2DeepScorePredictor
from omigami.spectra_matching.ms2deepscore.similarity_score_calculator import (
    MS2DeepScoreSimilarityScoreCalculator,
)
from omigami.test.conftest import ASSETS_DIR

redis_db = factories.redisdb("redis_nooproc")


@pytest.fixture
def positive_spectra_data(common_cleaned_data):
    spectra = [
        data for data in common_cleaned_data if data.get("ionmode") == "positive"
    ]
    return spectra


@pytest.fixture
def positive_spectra(positive_spectra_data):
    spectra = SpectrumProcessor().process_spectra(positive_spectra_data)
    return spectra


@pytest.fixture
def ms2deepscore_payload(loaded_data):
    spectra = [data for data in loaded_data if data["Ion_Mode"] == "Positive"]
    payload = {
        "data": [
            {
                "peaks_json": spectra[0]["peaks_json"],
                "Precursor_MZ": spectra[0]["Precursor_MZ"],
            },
            {
                "peaks_json": spectra[1]["peaks_json"],
                "Precursor_MZ": spectra[1]["Precursor_MZ"],
            },
        ],
        "parameters": {"n_best_spectra": 2, "include_metadata": ["Compound_name"]},
    }
    return payload


@pytest.fixture()
def ms2deepscore_model_path():
    return str(ASSETS_DIR / "ms2deepscore_model.hdf5")


@pytest.fixture()
def ms2deepscore_embedding(ms2deepscore_model_path):
    model = load_model(ms2deepscore_model_path)
    return MS2DeepScoreSimilarityScoreCalculator(model)


@pytest.fixture()
def ms2deepscore_predictor(ms2deepscore_embedding):
    predictor = MS2DeepScorePredictor(ion_mode="positive", run_id="2")
    predictor.model = ms2deepscore_embedding
    return predictor


@pytest.fixture()
def ms2deepscore_real_model_path():
    return str(
        ASSETS_DIR
        / "ms2deepscore"
        / "pretrained"
        / "MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5"
    )


@pytest.fixture()
def ms2deepscore_real_model(ms2deepscore_real_model_path):
    ms2deepscore_model = load_model(ms2deepscore_real_model_path)
    return ms2deepscore_model


@pytest.fixture()
def ms2deepscore_real_predictor(ms2deepscore_real_model):
    ms2deepscore_predictor = MS2DeepScorePredictor(ion_mode="positive", run_id="2")
    ms2deepscore_predictor._similarity_score_calculator = (
        MS2DeepScoreSimilarityScoreCalculator(ms2deepscore_real_model)
    )
    return ms2deepscore_predictor


@pytest.fixture()
def tanimoto_scores_path():
    return str(ASSETS_DIR / "ms2deepscore" / "to_train" / "tanimoto_scores.pkl")


@pytest.fixture()
def tanimoto_scores(tanimoto_scores_path):
    tanimoto_score = pd.read_pickle(tanimoto_scores_path, compression="gzip")
    return tanimoto_score


@pytest.fixture()
def binned_spectra_to_train():
    path = str(ASSETS_DIR / "ms2deepscore" / "to_train" / "binned_spectra.pkl")
    with open(path, "rb") as f:
        binned_spectra = pickle.load(f)
    return binned_spectra


@pytest.fixture
def binned_spectra_to_train_stored(redis_db, binned_spectra_to_train):
    pipe = redis_db.pipeline()
    for spectrum in binned_spectra_to_train:
        pipe.hset(
            f"{BINNED_SPECTRUM_HASHES}_positive",
            spectrum.metadata["spectrum_id"],
            pickle.dumps(spectrum),
        )
    pipe.execute()


@pytest.fixture()
def small_model_params(monkeypatch):
    smaller_params = {
        "batch_size": 2,
        "learning_rate": 0.001,
        "layer_base_dims": (300, 200, 100),
        "embedding_dim": 100,
        "dropout_rate": 0.2,
    }

    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.helper_classes.siamese_model_trainer,
        "SIAMESE_MODEL_PARAMS",
        smaller_params,
    )
