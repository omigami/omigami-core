import json

import numpy as np


class Embedding:
    def __init__(self, vector: np.ndarray, spectrum_id: str, n_decimals: int = None):
        self.vector = vector
        self.spectrum_id = spectrum_id
        self.n_decimals = n_decimals

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(
            vector=np.array(dct["vector"]),
            spectrum_id=dct["spectrum_id"],
            n_decimals=dct["n_decimals"],
        )


class EmbeddingJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Embedding):
            embedding_dict = {
                "spectrum_id": o.spectrum_id,
                "n_decimals": o.n_decimals,
                "vector": o.vector.tolist(),
            }
            return embedding_dict
        return json.JSONEncoder.default(self, o)
