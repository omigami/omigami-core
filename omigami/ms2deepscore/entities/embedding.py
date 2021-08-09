import numpy as np


class Embedding:
    def __init__(self, vector: np.ndarray, spectrum_id: str, inchikey: str):
        self.vector = vector
        self.spectrum_id = spectrum_id
        self.inchikey = inchikey
