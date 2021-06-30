import pytest
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from omigami.ms2deepscore.predictor import MS2DeepScorePredictor
from omigami.ms2deepscore.helper_classes.spectrum_processor import SpectrumProcessor
from omigami.test.conftest import ASSETS_DIR


@pytest.fixture
def positive_spectra_data(loaded_data):
    spectra = [data for data in loaded_data if data["Ion_Mode"] == "Positive"]
    return spectra


@pytest.fixture
def positive_spectra(positive_spectra_data):
    spectra = SpectrumProcessor().process_spectra(positive_spectra_data)
    return spectra


@pytest.fixture
def ms2deepscore_payload(positive_spectra_data):
    payload = {
        "data": [
            {
                "peaks_json": positive_spectra_data[0]["peaks_json"],
                "Precursor_MZ": positive_spectra_data[0]["Precursor_MZ"],
            },
            {
                "peaks_json": positive_spectra_data[1]["peaks_json"],
                "Precursor_MZ": positive_spectra_data[1]["Precursor_MZ"],
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
