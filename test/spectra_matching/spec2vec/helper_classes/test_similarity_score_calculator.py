import numpy as np
import pytest
from matchms import calculate_scores
from spec2vec import Spec2Vec

from omigami.spectra_matching.spec2vec.helper_classes.similarity_score_calculator import (
    Spec2VecSimilarityScoreCalculator,
)


@pytest.fixture()
def spec2vec_embeddings_similarity(word2vec_model):
    return Spec2VecSimilarityScoreCalculator(
        model=word2vec_model,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=25,
    )


@pytest.fixture()
def spec2vec_documents_similarity(word2vec_model):
    return Spec2Vec(
        model=word2vec_model,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=25,
    )


def test_pair(
    spec2vec_embeddings,
    documents_data,
    spec2vec_embeddings_similarity,
    spec2vec_documents_similarity,
):
    spectrum_ids = ["CCMSLIB00000072099", "CCMSLIB00000001778"]
    docs = [d for d in documents_data if d.metadata["spectrum_id"] in spectrum_ids]
    embs = [e for e in spec2vec_embeddings if e.spectrum_id in spectrum_ids]

    similarity_score_from_embeddings = spec2vec_embeddings_similarity.pair(*embs)
    similarity_score_from_documents = spec2vec_documents_similarity.pair(*docs)

    assert np.isclose(similarity_score_from_embeddings, similarity_score_from_documents)


@pytest.mark.xfail
def test_matrix(
    spec2vec_embeddings,
    documents_data,
    spec2vec_embeddings_similarity,
    spec2vec_documents_similarity,
):
    similarity_score_from_embeddings = spec2vec_embeddings_similarity.matrix(
        spec2vec_embeddings[:50], spec2vec_embeddings[50:]
    )

    similarity_score_from_documents = spec2vec_documents_similarity.matrix(
        documents_data[:50], documents_data[50:]
    )
    assert np.all(similarity_score_from_embeddings == similarity_score_from_documents)


@pytest.mark.xfail
def test_calculate_scores_with_spec2vec_embeddings(
    spec2vec_embeddings,
    documents_data,
    spec2vec_embeddings_similarity,
    spec2vec_documents_similarity,
):
    scores_from_embeddings = calculate_scores(
        spec2vec_embeddings[:50],
        spec2vec_embeddings[50:],
        spec2vec_embeddings_similarity,
        is_symmetric=False,
    )

    scores_from_documents = calculate_scores(
        documents_data[:50],
        documents_data[50:],
        spec2vec_documents_similarity,
        is_symmetric=False,
    )
    assert np.all(scores_from_embeddings.scores == scores_from_documents.scores)
    assert scores_from_embeddings.scores_by_query(spec2vec_embeddings[51])[0]
