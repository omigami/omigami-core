import numpy as np


class Embedding:
    # TODO delete this once MLOPS-453 is done, currently we cannot delete this
    #  because pickle test assets in `omigami/test/assets/ms2deepscore` still
    #  depends on this class

    def __init__(self, vector: np.ndarray, spectrum_id: str, inchikey: str):
        self.vector = vector
        self.spectrum_id = spectrum_id
        self.inchikey = inchikey
