from dataclasses import dataclass

from omigami.spectra_matching.entities.embedding import Embedding as BaseEmbedding


@dataclass
class Embedding(BaseEmbedding):
    # TODO: rename to `Spec2VecEmbedding`, when omigami/ms2deepscore/entities/embedding.py
    #  Embedding object is renamed. Remove `BaseEmbedding` from import
    n_decimals: int
