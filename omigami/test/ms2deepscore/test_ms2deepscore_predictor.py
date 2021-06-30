import os
import pandas as pd
import pytest
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

from omigami.ms2deepscore.predictor import MS2DeepScorePredictor
from omigami.test.conftest import ASSETS_DIR


def get_ms2deepscore_predictor():
    ms2deepscore_model_path = str(
        ASSETS_DIR
        / "ms2deepscore"
        / "pretrained"
        / "MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5"
    )
    ms2deepscore_model = MS2DeepScore(load_model(ms2deepscore_model_path))
    ms2deepscore_predictor = MS2DeepScorePredictor()
    ms2deepscore_predictor.model = ms2deepscore_model
    return ms2deepscore_predictor


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
@pytest.mark.skipif(
    not os.path.exists(
        str(
            ASSETS_DIR
            / "ms2deepscore"
            / "pretrained"
            / "MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5"
        )
    ),
    reason="MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5 is git ignored. Please "
    "download it from https://zenodo.org/record/4699356#.YNyD-2ZKhcA",
)
def test_predictions(
    ms2deepscore_payload,
    redis_full_setup,
    positive_spectra_data,
):
    ms2deepscore_predictor = get_ms2deepscore_predictor()
    scores = ms2deepscore_predictor.predict(
        data_input=ms2deepscore_payload,
        context="",
        mz_range=1,
    )

    assert isinstance(scores, dict)
    assert len(scores["spectrum-1"]) == 4


def test_parse_input(ms2deepscore_payload, ms2deepscore_predictor):
    data_input, parameters = ms2deepscore_predictor._parse_input(ms2deepscore_payload)

    assert len(data_input) == 2
    assert "peaks_json" in data_input[0]
    assert "Precursor_MZ" in data_input[0]
    assert parameters["n_best"] == 2


@pytest.mark.skipif(
    not os.path.exists(
        str(
            ASSETS_DIR
            / "ms2deepscore"
            / "pretrained"
            / "MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5"
        )
    ),
    reason="MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5 is git ignored. Please "
    "download it from https://zenodo.org/record/4699356#.YNyD-2ZKhcA",
)
def test_get_best_matches(positive_spectra):
    ms2deepscore_predictor = get_ms2deepscore_predictor()
    n_best_spectra = 2
    best_matches = {}
    for query in positive_spectra[:2]:
        input_best_matches = ms2deepscore_predictor._calculate_best_matches(
            positive_spectra,
            query,
            n_best_spectra=n_best_spectra,
        )
        best_matches[query.metadata["spectrum_id"]] = input_best_matches

    for query, (best_match_id, best_match) in zip(
        positive_spectra, best_matches.items()
    ):
        assert len(best_match) == n_best_spectra
        assert query.metadata["spectrum_id"] == best_match_id
        assert "score" in pd.DataFrame(best_match).T.columns
