import os
from copy import deepcopy

import pandas as pd
import pytest

from omigami.test.conftest import ASSETS_DIR


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
    ms2deepscore_real_predictor,
):
    scores = ms2deepscore_real_predictor.predict(
        data_input=ms2deepscore_payload,
        context="",
        mz_range=1,
    )

    assert isinstance(scores, dict)
    scores_df = pd.DataFrame(scores["spectrum-0"]).T
    assert scores_df["score"].between(0, 1).all()
    assert all([len(value) == 2] for value in scores.values())


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
def test_predictions_1_best(
    ms2deepscore_payload,
    redis_full_setup,
    ms2deepscore_real_predictor,
):
    ms2deepscore_payload_copy = deepcopy(ms2deepscore_payload)
    ms2deepscore_payload_copy["parameters"]["n_best"] = 1

    scores = ms2deepscore_real_predictor.predict(
        data_input=ms2deepscore_payload,
        context="",
        mz_range=1,
    )

    assert isinstance(scores, dict)
    scores_df = pd.DataFrame(scores["spectrum-0"]).T
    assert scores_df["score"].between(0, 1).all()
    assert all([len(value) == 1] for value in scores.values())


def test_parse_input(ms2deepscore_payload, ms2deepscore_predictor):
    data_input, parameters = ms2deepscore_predictor._parse_input(ms2deepscore_payload)

    assert len(data_input) == 2
    assert "peaks_json" in data_input[0]
    assert "Precursor_MZ" in data_input[0]
    assert parameters["n_best_spectra"] == 2


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
def test_get_best_matches(embeddings_from_real_predictor, ms2deepscore_real_predictor):
    n_best_spectra = 2
    best_matches = ms2deepscore_real_predictor._calculate_best_matches(
        embeddings_from_real_predictor,
        embeddings_from_real_predictor,
        n_best_spectra=n_best_spectra,
    )

    for query, best_match in zip(embeddings_from_real_predictor, best_matches.values()):
        assert len(best_match) == n_best_spectra
        assert query.spectrum_id == list(best_match.keys())[0]
        assert "score" in pd.DataFrame(best_match).T.columns
