import pytest
from matchms import calculate_scores
from ms2deepscore import MS2DeepScore
import numpy as np
from omigami.ms2deepscore.helper_classes.ms2deepscore_binned_spectrum import (
    MS2DeepScoreBinnedSpectrum,
)


@pytest.fixture()
def ms2deepscore_binned_spectrum_similarity(ms2deepscore_real_model):
    return MS2DeepScoreBinnedSpectrum(ms2deepscore_real_model)


@pytest.fixture()
def ms2deepscore_spectrum_similarity(ms2deepscore_real_model):
    return MS2DeepScore(ms2deepscore_real_model)


def test_pair(
    positive_spectra,
    binned_spectra,
    ms2deepscore_spectrum_similarity,
    ms2deepscore_binned_spectrum_similarity,
):
    spectrum_ids = ["CCMSLIB00000001577", "CCMSLIB00000001567"]
    spectra = [
        spectrum
        for spectrum in positive_spectra
        if spectrum.metadata["spectrum_id"] in spectrum_ids
    ]
    binned_spectra = [
        binned_spectrum
        for binned_spectrum in binned_spectra
        if binned_spectrum.metadata["spectrum_id"] in spectrum_ids
    ]

    similarity_score_from_spectrum = ms2deepscore_spectrum_similarity.pair(*spectra)
    similarity_score_from_binned_spectrum = (
        ms2deepscore_binned_spectrum_similarity.pair(*binned_spectra)
    )

    assert np.isclose(
        similarity_score_from_spectrum, similarity_score_from_binned_spectrum
    )


def test_matrix(
    positive_spectra,
    binned_spectra,
    ms2deepscore_spectrum_similarity,
    ms2deepscore_binned_spectrum_similarity,
):

    score_from_spectrum = ms2deepscore_spectrum_similarity.matrix(
        positive_spectra[:10], positive_spectra[10:12]
    )
    score_from_binned_spectrum = ms2deepscore_binned_spectrum_similarity.matrix(
        binned_spectra[:10], binned_spectra[10:12]
    )
    assert np.all(score_from_spectrum == score_from_binned_spectrum)


def test_calculate_scores_with_ms2deepscore_binned_spectrum(
    positive_spectra,
    binned_spectra,
    ms2deepscore_spectrum_similarity,
    ms2deepscore_binned_spectrum_similarity,
):
    scores_from_embeddings = calculate_scores(
        positive_spectra[:50],
        positive_spectra[50:],
        ms2deepscore_spectrum_similarity,
        is_symmetric=False,
    )

    scores_from_documents = calculate_scores(
        binned_spectra[:50],
        binned_spectra[50:],
        ms2deepscore_binned_spectrum_similarity,
        is_symmetric=False,
    )
    assert np.all(scores_from_embeddings.scores == scores_from_documents.scores)
