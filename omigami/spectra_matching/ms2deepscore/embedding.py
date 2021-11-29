from dataclasses import dataclass

from omigami.spectra_matching.entities.embedding import Embedding


@dataclass
class MS2DeepScoreEmbedding(Embedding):
    inchikey: str
