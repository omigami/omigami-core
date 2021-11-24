from dataclasses import dataclass

import numpy as np


@dataclass
class Embedding:
    vector: np.ndarray
    spectrum_id: str
