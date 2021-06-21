import os
from pathlib import Path

import pytest
from matchms.importing import load_from_mgf
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

from omigami.ms2deepscore.predictor import Predictor
from omigami.test.conftest import ASSETS_DIR

os.chdir(Path(__file__).parents[3])


@pytest.fixture
def spectra():
    spectra = list(load_from_mgf(str(ASSETS_DIR / "spectra.mgf")))
    return spectra


@pytest.fixture
def ms2deepscore_payload(spectra):
    payload = {
        "data": [
            {
                "peaks_json": str(
                    [
                        [mz, intensity]
                        for mz, intensity in zip(
                            spectrum.peaks.mz, spectrum.peaks.intensities
                        )
                    ]
                )
            }
            for spectrum in spectra
        ]
    }
    return payload


@pytest.fixture
def payload_identical_spectra(spectra):
    spectra = [spectra[0], spectra[0]]
    payload = {
        "data": [
            {
                "peaks_json": str(
                    [
                        [mz, intensity]
                        for mz, intensity in zip(
                            spectrum.peaks.mz, spectrum.peaks.intensities
                        )
                    ]
                )
            }
            for spectrum in spectra
        ]
    }
    return payload


@pytest.fixture()
def ms2deepscore_model():
    path = str(ASSETS_DIR / "ms2deepscore_model.hdf5")
    model = load_model(path)
    model = MS2DeepScore(model)
    return Predictor(model=model)


def test_predictions(ms2deepscore_model, ms2deepscore_payload, spectra):
    score = ms2deepscore_model.predict(
        data_input=ms2deepscore_payload,
        context="",
    )

    assert type(score) == float
    assert 0 <= score <= 1


def test_predictions_identical_spectra(ms2deepscore_model, payload_identical_spectra):
    score = ms2deepscore_model.predict(
        data_input=payload_identical_spectra,
        context="",
    )

    assert score == 1


def test_parse_input(ms2deepscore_payload, ms2deepscore_model):
    data_input = ms2deepscore_model._parse_input(ms2deepscore_payload)

    assert len(data_input) == 2
    assert "peaks_json" in data_input[0]
