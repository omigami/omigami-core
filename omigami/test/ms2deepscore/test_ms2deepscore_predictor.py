import os
import pandas as pd
import pytest
from matchms import Spectrum


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_predictions(
    ms2deepscore_payload, ms2deepscore_predictor, redis_full_setup, spectra
):
    scores = ms2deepscore_predictor.predict(
        data_input=ms2deepscore_payload,
        context="",
        mz_range=20,
    )

    assert isinstance(scores, dict)
    assert len(scores["spectrum-0"]) == 3


def test_parse_input(ms2deepscore_payload, ms2deepscore_predictor):
    data_input, parameters = ms2deepscore_predictor._parse_input(ms2deepscore_payload)

    assert len(data_input) == 2
    assert "intensities" in data_input[0]
    assert "mz" in data_input[0]
    assert parameters["n_best"] == 2


def test_clean_spectra(ms2deepscore_predictor):
    data_input = [
        {"intensities": [80.060677, 337.508301], "mz": [157.0, 230.0]},
        {"intensities": [81.060677, 339.508301], "mz": [158.0, 240.0]},
    ]
    spectra = ms2deepscore_predictor._clean_spectra(data_input)

    assert isinstance(spectra[0], Spectrum)


def test_get_best_matches(ms2deepscore_predictor, spectra):
    n_best_spectra = 2
    best_matches = {}
    for query in spectra[:2]:
        input_best_matches = ms2deepscore_predictor._calculate_best_matches(
            spectra,
            query,
            n_best_spectra=n_best_spectra,
        )
        best_matches[query.metadata["spectrum_id"]] = input_best_matches

    for query, (best_match_id, best_match) in zip(spectra, best_matches.items()):
        assert len(best_match) == n_best_spectra
        assert query.metadata["spectrum_id"] == best_match_id
        assert "score" in pd.DataFrame(best_match).T.columns
