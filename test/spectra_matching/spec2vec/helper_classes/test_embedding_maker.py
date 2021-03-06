import numpy as np
import pytest

from omigami.spectra_matching.spec2vec.helper_classes.embedding_maker import (
    EmbeddingMaker,
    EmbeddingMakerError,
)


def test_check_n_decimals_success(documents_data):
    em = EmbeddingMaker(n_decimals=1)

    em._check_n_decimals(documents_data[0])


def test_check_n_decimals_fail(documents_data):
    em = EmbeddingMaker(n_decimals=2)

    with pytest.raises(EmbeddingMakerError):
        em._check_n_decimals(documents_data[0])


def test_make_embeddings(documents_data, word2vec_model):
    em = EmbeddingMaker(n_decimals=1)

    res = em.make_embedding(
        model=word2vec_model,
        document=documents_data[0],
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5.0,
    )
    assert isinstance(res.vector, np.ndarray)
