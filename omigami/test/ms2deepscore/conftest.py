import pickle

import pandas as pd
import pytest
from ms2deepscore.models import load_model
from pytest_redis import factories

from omigami.ms2deepscore.config import BINNED_SPECTRUM_HASHES
from omigami.ms2deepscore.helper_classes.ms2deepscore_binned_spectrum import (
    MS2DeepScoreBinnedSpectrum,
)
from omigami.ms2deepscore.helper_classes.spectrum_processor import (
    SpectrumProcessor,
)
from omigami.ms2deepscore.predictor import MS2DeepScorePredictor
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
        "parameters": {"n_best": 2, "include_metadata": ["Compound_name"]},
    }
    return payload


@pytest.fixture()
def ms2deepscore_model_path():
    return str(ASSETS_DIR / "ms2deepscore_model.hdf5")


@pytest.fixture()
def ms2deepscore_model(ms2deepscore_model_path):
    model = load_model(ms2deepscore_model_path)
    return MS2DeepScoreBinnedSpectrum(model)


@pytest.fixture()
def ms2deepscore_spectrum_similarity(ms2deepscore_model_path):
    model = load_model(ms2deepscore_model_path)
    return MS2DeepScoreBinnedSpectrum(model)


@pytest.fixture()
def ms2deepscore_predictor(ms2deepscore_model):
    predictor = MS2DeepScorePredictor()
    predictor.model = ms2deepscore_model

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
    ms2deepscore_predictor = MS2DeepScorePredictor()
    ms2deepscore_predictor.model = MS2DeepScoreBinnedSpectrum(ms2deepscore_real_model)
    return ms2deepscore_predictor


@pytest.fixture()
def binned_spectra_from_real_predictor(ms2deepscore_real_predictor, positive_spectra):
    binned_spectra = ms2deepscore_real_predictor.model.model.spectrum_binner.transform(
        positive_spectra
    )
    return [
        binned_spectrum.set("spectrum_id", positive_spectra[i].metadata["spectrum_id"])
        for i, binned_spectrum in enumerate(binned_spectra)
    ]


@pytest.fixture(scope="module")
def cleaned_data_ms2deep_score():
    path = str(ASSETS_DIR / "ms2deepscore" / "SMALL_GNPS_cleaned.pickle")
    with open(path, "rb") as handle:
        cleaned_data = pickle.load(handle)
    return cleaned_data


@pytest.fixture()
def tanimoto_scores_path():
    return str(ASSETS_DIR / "ms2deepscore" / "to_train" / "tanimoto_scores.pkl")


@pytest.fixture()
def tanimoto_scores(tanimoto_scores_path):
    tanimoto_score = pd.read_pickle(tanimoto_scores_path, compression="gzip")
    return tanimoto_score


@pytest.fixture()
def fitted_spectrum_binner_path():
    return str(ASSETS_DIR / "ms2deepscore" / "to_train" / "fitted_spectrum_binner.pkl")


@pytest.fixture()
def fitted_spectrum_binner(fitted_spectrum_binner_path):
    with open(fitted_spectrum_binner_path, "rb") as f:
        spectrum_binner = pickle.load(f)
    return spectrum_binner


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
            f"{BINNED_SPECTRUM_HASHES}",
            spectrum.metadata["spectrum_id"],
            pickle.dumps(spectrum),
        )
    pipe.execute()
