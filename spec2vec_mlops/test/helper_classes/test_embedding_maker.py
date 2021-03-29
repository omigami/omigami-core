import numpy as np

from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker


def test_make_embeddings(documents_data, word2vec_model):
    em = EmbeddingMaker()

    for doc in documents_data:
        res = em.make_embeddings(word2vec_model, doc, 5.0)
        assert isinstance(res, np.ndarray)
