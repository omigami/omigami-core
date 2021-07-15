import os

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
    positive_spectra_data,
    ms2deepscore_real_predictor,
):
    scores = ms2deepscore_real_predictor.predict(
        data_input=ms2deepscore_payload,
        context="",
        mz_range=1,
    )

    assert isinstance(scores, dict)
    scores_df = pd.DataFrame(scores["spectrum-1"]).T
    assert scores_df["score"].between(0, 1).all()


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
def test_get_best_matches(
    binned_spectra_from_real_predictor, ms2deepscore_real_predictor
):
    positive_spectra_ids = [
        spectrum.metadata["spectrum_id"]
        for spectrum in binned_spectra_from_real_predictor
    ]
    spectrum_ids_near_queries_mz = [
        positive_spectra_ids,
        positive_spectra_ids,
    ]
    n_best_spectra = 2
    best_matches = {}
    for i, query in enumerate(binned_spectra_from_real_predictor[:2]):
        input_best_matches = ms2deepscore_real_predictor._calculate_best_matches(
            binned_spectra_from_real_predictor,
            spectrum_ids_near_queries_mz[i],
            query,
            n_best_spectra=n_best_spectra,
        )
        best_matches[query.metadata["spectrum_id"]] = input_best_matches

    for query, (best_match_id, best_match) in zip(
        binned_spectra_from_real_predictor, best_matches.items()
    ):
        assert len(best_match) == n_best_spectra
        assert query.metadata["spectrum_id"] == best_match_id
        assert "score" in pd.DataFrame(best_match).T.columns
