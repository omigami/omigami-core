from dataclasses import dataclass

from omigami.spectra_matching.entities.embedding import Embedding


@dataclass
class Spec2VecEmbedding(Embedding):
    n_decimals: int
