from typing import Optional

import numpy as np
from matchms import Spectrum


class CleanedSpectrum(Spectrum):
    def __init__(
        self,
        mz: np.array,
        intensities: np.array,
        metadata: Optional[dict] = None,
        spectrum_id: str = None,
    ):
        super().__init__(mz, intensities, metadata)
        self.spectrum_id = spectrum_id

    @property
    def spectrum_id(self) -> str:
        return self.spectrum_id
