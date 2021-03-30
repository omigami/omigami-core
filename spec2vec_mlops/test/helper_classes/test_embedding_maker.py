import numpy as np

from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker


def test_make_embeddings(documents_data, word2vec_model):
    em = EmbeddingMaker()

    res = em.make_embedding(
        model=word2vec_model,
        document=documents_data[0],
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5.0,
    )
    assert isinstance(res, np.ndarray)
