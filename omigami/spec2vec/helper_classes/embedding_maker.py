from typing import Union

from gensim.models import Word2Vec
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector

from omigami.spec2vec.entities.embedding import Embedding


class EmbeddingMakerError(Exception):
    pass


class EmbeddingMaker:
    def __init__(self, n_decimals: int = 2):
        self.n_decimals = n_decimals

    def make_embedding(
        self,
        model: Word2Vec,
        document: Union[SpectrumDocument],
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
        return Embedding(vector, document.get("spectrum_id"), self.n_decimals)

    def _check_n_decimals(
        self,
        document: SpectrumDocument,
    ):
        if self.n_decimals != document.n_decimals:
            raise EmbeddingMakerError(
                "Decimal rounding of input data does not agree with model vocabulary."
            )
