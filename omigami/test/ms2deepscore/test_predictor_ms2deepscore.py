import os
from pathlib import Path


os.chdir(Path(__file__).parents[3])


def test_predictions(ms2deepscore_payload, ms2deepscore_predictor):
    score = ms2deepscore_predictor.predict(
        data_input=ms2deepscore_payload,
        context="",
    )

    assert type(score) == float
    assert 0 <= score <= 1


def test_predictions_identical_spectra(
    ms2deepscore_predictor, payload_identical_spectra
):
    score = ms2deepscore_predictor.predict(
        data_input=payload_identical_spectra,
        context="",
    )

    assert score == 1


def test_parse_input(ms2deepscore_payload, ms2deepscore_predictor):
    data_input = ms2deepscore_predictor._parse_input(ms2deepscore_payload)

    assert len(data_input) == 2
    assert "intensities" in data_input[0]
    assert "mz" in data_input[0]
