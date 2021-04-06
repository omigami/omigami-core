from typing import Union

from gensim.models import Word2Vec
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector

from spec2vec_mlops.helper_classes.embedding import Embedding
from spec2vec_mlops.helper_classes.exception import EmbeddingMakerError


class EmbeddingMaker:
    def __init__(self, n_decimals: int = 2):
        self.n_decimals = n_decimals

    @staticmethod
    def make_embedding(
        model: Word2Vec,
        document: SpectrumDocument,
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

    def _check_n_decimals(
        self,
        document: SpectrumDocument,
    ):
        if self.n_decimals != document.n_decimals:
            raise EmbeddingMakerError(
                "Decimal rounding of input data does not agree with model vocabulary."
            )
