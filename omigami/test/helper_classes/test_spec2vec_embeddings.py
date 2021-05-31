import numpy as np
import pytest
from matchms import calculate_scores
from spec2vec import Spec2Vec

from omigami.helper_classes.spec2vec_embeddings import Spec2VecEmbeddings


@pytest.fixture()
def spec2vec_embeddings_similarity(word2vec_model):
    return Spec2VecEmbeddings(
        model=word2vec_model,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5.0,
    )


@pytest.fixture()
def spec2vec_documents_similarity(word2vec_model):
    return Spec2Vec(
        model=word2vec_model,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5.0,
    )


def test_pair(
    embeddings,
    documents_data,
    spec2vec_embeddings_similarity,
    spec2vec_documents_similarity,
):
    similarity_score_from_embeddings = spec2vec_embeddings_similarity.pair(
        embeddings[0], embeddings[1]
    )
    similarity_score_from_documents = spec2vec_documents_similarity.pair(
        documents_data[0], documents_data[1]
    )
    assert similarity_score_from_embeddings == similarity_score_from_documents


@pytest.mark.xfail
def test_matrix(
    embeddings,
    documents_data,
    spec2vec_embeddings_similarity,
    spec2vec_documents_similarity,
):
    similarity_score_from_embeddings = spec2vec_embeddings_similarity.matrix(
        embeddings[:50], embeddings[50:]
    )

    similarity_score_from_documents = spec2vec_documents_similarity.matrix(
        documents_data[:50], documents_data[50:]
    )
    assert np.all(similarity_score_from_embeddings == similarity_score_from_documents)


@pytest.mark.xfail
def test_calculate_scores_with_spec2vec_embeddings(
    embeddings,
    documents_data,
    spec2vec_embeddings_similarity,
    spec2vec_documents_similarity,
):
    scores_from_embeddings = calculate_scores(
        embeddings[:50],
        embeddings[50:],
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
    assert scores_from_embeddings.scores_by_query(embeddings[51])[0]
