from typing import Union

import numpy as np
from gensim.models import Word2Vec
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector


class EmbeddingMaker:
    @staticmethod
    def make_embeddings(
        model: Word2Vec,
        document: SpectrumDocument,
        allowed_missing_percentage: Union[float, int] = 5.0,
        intensity_weighting_power: Union[float, int] = 0.5,
    ) -> np.ndarray:
        embedding = calc_vector(
            model=model,
            document=document,
            intensity_weighting_power=intensity_weighting_power,
            allowed_missing_percentage=allowed_missing_percentage,
        )
        return embedding
