import numpy as np
from numpy.testing import assert_almost_equal

from omigami.spectra_matching.ms2deepscore.util import _cosine_similarity


def test_cosine_similarity():
    vector1 = np.array([1, 1, 0, 0])
    vector2 = np.array([1, 1, 1, 1])

    res = _cosine_similarity(vector1, vector2)

    assert_almost_equal(res, 0.707, decimal=3)
