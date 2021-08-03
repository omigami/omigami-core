import numpy as np
from omigami.ms2deepscore.entities.embedding import Embedding
from omigami.ms2deepscore.helper_classes.embedding_maker import EmbeddingMaker


def test_make_embedding(binned_spectra, ms2deepscore_real_model_path):
    maker = EmbeddingMaker()
    embedding = maker.make_embedding(ms2deepscore_real_model_path, binned_spectra[0])
    assert isinstance(embedding, Embedding)
    assert isinstance(embedding.vector, np.ndarray)
