from typing import Optional

from matchms import Spectrum
from spec2vec import SpectrumDocument


class SpectrumDocumentData:
    """Spectrum info and Document to be stored in data store."""

    def __init__(self, spectrum: Spectrum, n_decimals: int):
        self.spectrum = spectrum
        self.n_decimals = n_decimals

    @property
    def spectrum_id(self) -> str:
        return self.spectrum.metadata["spectrum_id"]

    @property
    def ionmode(self) -> str:
        return self.spectrum.metadata["ionmode"]

    @property
    def document(self) -> Optional[SpectrumDocument]:
        try:
            document = SpectrumDocument(self.spectrum, n_decimals=self.n_decimals)
        except AssertionError:
            return
        return document

    @property
    def precursor_mz(self) -> float:
        return self.spectrum.metadata["precursor_mz"]
