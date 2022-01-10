import numpy as np
from numpy.testing import assert_almost_equal

from omigami.spectra_matching.util import (
    cosine_similarity,
    cosine_similarity_matrix,
)
from spec2vec.vector_operations import (
    cosine_similarity as spec2vec_cosine_similarity,
    cosine_similarity_matrix as spec2vec_cosine_similarity_matrix,
)


def test_cosine_similarity():
    vector1 = np.array([1, 1, 0, 0])
    vector2 = np.array([1, 1, 1, 1])

    expected = spec2vec_cosine_similarity(vector1, vector2)
    res = cosine_similarity(vector1, vector2)

    assert_almost_equal(res, expected)


def test_cosine_similarity_matrix():
    vectors_1 = np.array([[1, 1, 0, 0], [1, 0, 1, 1]])
    vectors_2 = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])

    expected = spec2vec_cosine_similarity_matrix(vectors_1, vectors_2)
    res = cosine_similarity_matrix(vectors_1, vectors_2)

    assert np.allclose(expected, res)
