import os
import numpy as np
from pathlib import Path


os.chdir(Path(__file__).parents[3])


def test_predictions(ms2deepscore_payload, ms2deepscore_predictor):
    score = ms2deepscore_predictor.predict(
        data_input=ms2deepscore_payload,
        context="",
    )

    assert type(score["score"]) == float
    assert 0 <= score["score"] <= 1


def test_predictions_identical_spectra(
    ms2deepscore_predictor, payload_identical_spectra
):
    score = ms2deepscore_predictor.predict(
        data_input=payload_identical_spectra,
        context="",
    )

    assert score["score"] == 1


def test_parse_input(ms2deepscore_payload, ms2deepscore_predictor):
    data_input = ms2deepscore_predictor._parse_input(ms2deepscore_payload)

    assert len(data_input) == 2
    assert "intensities" in data_input[0]
    assert "mz" in data_input[0]


def test_clean_spectra(ms2deepscore_predictor):
    data_input = [
        {"intensities": "[80.060677, 337.508301]", "mz": "[157.0, 230.0]"},
        {"intensities": "[81.060677, 339.508301]", "mz": "[158.0, 240.0]"},
    ]
    spectra = ms2deepscore_predictor._clean_spectra(data_input)
    assert isinstance(spectra[0]["intensities"], np.ndarray)
    assert isinstance(spectra[1]["mz"], np.ndarray)
