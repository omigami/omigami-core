from typing import Union

from gensim.models import Word2Vec
from spec2vec.vector_operations import calc_vector

from spec2vec_mlops.entities.embedding import Embedding
from spec2vec_mlops.helper_classes.storer_classes import FeastSpectrumDocument
from spec2vec_mlops.helper_classes.exception import EmbeddingMakerError


class EmbeddingMaker:
    def __init__(self, n_decimals: int = 2):
        self.n_decimals = n_decimals

    def make_embedding(
        self,
        model: Word2Vec,
        document: FeastSpectrumDocument,
        intensity_weighting_power: Union[float, int] = None,
        allowed_missing_percentage: Union[float, int] = None,
    ) -> Embedding:
        self._check_n_decimals(document)
        vector = calc_vector(
            model=model,
            document=document,
            intensity_weighting_power=intensity_weighting_power,
            allowed_missing_percentage=allowed_missing_percentage,
        )
        return Embedding(vector, document.metadata["spectrum_id"], self.n_decimals)

    def _check_n_decimals(
        self,
        document: FeastSpectrumDocument,
    ):
        if self.n_decimals != document.n_decimals:
            raise EmbeddingMakerError(
                "Decimal rounding of input data does not agree with model vocabulary."
            )
