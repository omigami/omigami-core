from copy import deepcopy

import pytest
from matchms.importing import load_from_mgf
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

from omigami.ms2deepscore.predictor import Predictor
from omigami.test.conftest import ASSETS_DIR


@pytest.fixture
def spectra():
    spectra = list(load_from_mgf(str(ASSETS_DIR / "spectra.mgf")))
    return spectra


@pytest.fixture
def ms2deepscore_payload(spectra):
    reference = spectra[0]
    query = spectra[1]
    payload = {
        "data": [
            {
                "intensities": reference.peaks.intensities,
                "mz": reference.peaks.mz,
            },
            {
                "intensities": query.peaks.intensities,
                "mz": query.peaks.mz,
            },
        ],
    }
    return payload


@pytest.fixture
def payload_identical_spectra(spectra):
    reference = spectra[0]
    payload = {
        "data": [
            {
                "intensities": reference.peaks.intensities,
                "mz": reference.peaks.mz,
            },
            {
                "intensities": reference.peaks.intensities,
                "mz": reference.peaks.mz,
            },
        ],
    }
    return payload


@pytest.fixture()
def ms2deepscore_model():
    path = str(ASSETS_DIR / "ms2deepscore_model.hdf5")
    model = load_model(path)
    return MS2DeepScore(deepcopy(model))


@pytest.fixture()
def ms2deepscore_predictor(ms2deepscore_model):
    return Predictor(model=ms2deepscore_model)
