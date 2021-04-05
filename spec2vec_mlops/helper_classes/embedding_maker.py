from typing import Union

from gensim.models import Word2Vec
from spec2vec.vector_operations import calc_vector

from spec2vec_mlops.helper_classes.storer_classes import FeastSpectrumDocument
from spec2vec_mlops.helper_classes.embedding import Embedding


class EmbeddingMaker:
    @staticmethod
    def make_embedding(
        model: Word2Vec,
        document: FeastSpectrumDocument,
        intensity_weighting_power: Union[float, int] = None,
        allowed_missing_percentage: Union[float, int] = None,
    ) -> Embedding:
        vector = calc_vector(
            model=model,
            document=document,
            intensity_weighting_power=intensity_weighting_power,
            allowed_missing_percentage=allowed_missing_percentage,
        )
        return Embedding(vector, document.metadata["spectrum_id"])
