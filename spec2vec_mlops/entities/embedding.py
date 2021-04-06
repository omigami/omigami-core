import numpy as np


class Embedding:
    def __init__(self, vector: np.ndarray, spectrum_id: str, n_decimals: int):
        self.vector = vector
        self.spectrum_id = spectrum_id
        self.n_decimals = n_decimals
