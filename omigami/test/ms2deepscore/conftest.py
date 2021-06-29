import pytest
from matchms.importing.load_from_json import as_spectrum
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from omigami.ms2deepscore.predictor import MS2DeepScorePredictor
from omigami.test.conftest import ASSETS_DIR


@pytest.fixture
def positive_spectra(loaded_data):
    spectra = [
        as_spectrum(data) for data in loaded_data if data["Ion_Mode"] == "Positive"
    ]
    return spectra


@pytest.fixture
def ms2deepscore_payload(positive_spectra):
    payload = {
        "data": [
            {
                "intensities": list(positive_spectra[0].peaks.intensities),
                "mz": list(positive_spectra[0].peaks.mz),
                "Precursor_MZ": positive_spectra[0].metadata["precursor_mz"],
            },
            {
                "intensities": list(positive_spectra[1].peaks.intensities),
                "mz": list(positive_spectra[1].peaks.mz),
                "Precursor_MZ": positive_spectra[1].metadata["precursor_mz"],
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
    return MS2DeepScore(model)


@pytest.fixture()
def ms2deepscore_predictor(ms2deepscore_model):
    predictor = MS2DeepScorePredictor()
    predictor.model = ms2deepscore_model

    return predictor
