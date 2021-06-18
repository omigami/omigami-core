import os
from pathlib import Path

import pytest
from matchms.importing import load_from_mgf
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

from omigami.ms2deep.predictor import Predictor
from omigami.test.conftest import ASSETS_DIR

os.chdir(Path(__file__).parents[3])


@pytest.fixture
def spectra():
    spectra = list(load_from_mgf(str(ASSETS_DIR / "spectra.mgf")))
    return spectra


@pytest.fixture
def payload(spectra):
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


@pytest.fixture()
def model():
    path = str(ASSETS_DIR / "ms2deepscore_model.hdf5")
    model = load_model(path)
    model = MS2DeepScore(model)
    return Predictor(model=model)


@pytest.mark.skipif(
    not os.path.exists(str(ASSETS_DIR / "ms2deepscore_model.hdf5")),
    reason="ms2deepscore_model.hdf5 is git ignored",
)
def test_predictions(model, payload):
    score = model.predict(
        data_input=payload,
        context="",
    )

    assert type(score) == float
    assert 0 <= score <= 1


@pytest.mark.skipif(
    not os.path.exists(str(ASSETS_DIR / "ms2deepscore_model.hdf5")),
    reason="ms2deepscore_model.hdf5 is git ignored",
)
def test_score(model, spectra):
    score = model.score(spectra[0], spectra[1])
    assert type(score) == float
    assert 0 <= score <= 1

    score = model.score(spectra[1], spectra[1])
    assert score == 1


def test_parse_input(payload, model):
    data_input = model._parse_input(payload)

    assert len(data_input) == 2
    assert "intensities" in data_input[0]
    assert "mz" in data_input[0]
