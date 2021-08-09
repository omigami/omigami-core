import os

import numpy as np
import pytest
from matchms import calculate_scores
from ms2deepscore import MS2DeepScore
from omigami.ms2deepscore.helper_classes.ms2deepscore_embedding import (
    MS2DeepScoreEmbedding,
)
from omigami.test.conftest import ASSETS_DIR

pytestmark = pytest.mark.skipif(
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


@pytest.fixture()
def ms2deepscore_embedding_similarity(ms2deepscore_real_model):
    return MS2DeepScoreEmbedding(ms2deepscore_real_model)


@pytest.fixture()
def ms2deepscore_spectrum_similarity(ms2deepscore_real_model):
    return MS2DeepScore(ms2deepscore_real_model)


def test_pair(
    positive_spectra,
    embeddings_from_real_predictor,
    ms2deepscore_spectrum_similarity,
    ms2deepscore_embedding_similarity,
):
    spectrum_ids = ["CCMSLIB00000001577", "CCMSLIB00000001647"]
    spectra = [
        spectrum
        for spectrum in positive_spectra
        if spectrum.metadata["spectrum_id"] in spectrum_ids
    ]
    embeddings = [
        binned_spectrum
        for binned_spectrum in embeddings_from_real_predictor
        if binned_spectrum.spectrum_id in spectrum_ids
    ]

    similarity_score_from_spectrum = ms2deepscore_spectrum_similarity.pair(*spectra)
    similarity_score_from_embedding = ms2deepscore_embedding_similarity.pair(
        *embeddings
    )

    assert np.isclose(similarity_score_from_spectrum, similarity_score_from_embedding)


def test_matrix(
    positive_spectra,
    embeddings_from_real_predictor,
    ms2deepscore_spectrum_similarity,
    ms2deepscore_embedding_similarity,
):

    score_from_spectrum = ms2deepscore_spectrum_similarity.matrix(
        positive_spectra[:10], positive_spectra[10:12]
    )
    score_from_embeddings = ms2deepscore_embedding_similarity.matrix(
        embeddings_from_real_predictor[:10],
        embeddings_from_real_predictor[10:12],
    )
    assert np.all(score_from_spectrum == score_from_embeddings)


def test_calculate_scores_with_ms2deepscore_binned_spectrum(
    positive_spectra,
    embeddings_from_real_predictor,
    ms2deepscore_spectrum_similarity,
    ms2deepscore_embedding_similarity,
):
    scores_from_spectra = calculate_scores(
        positive_spectra[:50],
        positive_spectra[50:],
        ms2deepscore_spectrum_similarity,
        is_symmetric=False,
    )

    scores_from_embeddings = calculate_scores(
        embeddings_from_real_predictor[:50],
        embeddings_from_real_predictor[50:],
        ms2deepscore_embedding_similarity,
        is_symmetric=False,
    )
    assert np.all(scores_from_spectra.scores == scores_from_embeddings.scores)
