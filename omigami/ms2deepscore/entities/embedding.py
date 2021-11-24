from dataclasses import dataclass

from omigami.spectra_matching.entities.embedding import Embedding as BaseEmbedding


@dataclass
class Embedding(BaseEmbedding):
    # TODO rename to `MS2DeepScoreEmbedding` once MLOPS-453 is done
    #  currently name as kept as it is, not to have conflicts
    #  with existing pickle test assets under omigami/test/assets/ms2deepscore
    #  Remove `BaseEmbedding` from import
    inchikey: str
