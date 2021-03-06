from typing import Optional, List

from matchms import Spectrum
from spec2vec import SpectrumDocument


class SpectrumDocumentData:
    """Spectrum info and Document to be stored in data store."""

    def __init__(self, spectrum: Spectrum, n_decimals: int):
        self._document = SpectrumDocument(spectrum, n_decimals=n_decimals)

    @property
    def spectrum_id(self) -> str:
        return self._document.get("spectrum_id")

    @property
    def document(self) -> Optional[SpectrumDocument]:
        return self._document

    @property
    def n_decimals(self) -> int:
        return self._document.n_decimals

    @property
    def weights(self) -> List[float]:
        return self._document.weights

    @property
    def words(self) -> List[str]:
        return self._document.words
