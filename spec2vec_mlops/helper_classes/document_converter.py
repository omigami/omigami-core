from matchms import Spectrum
from spec2vec import SpectrumDocument


class DocumentConverter:
    @staticmethod
    def convert_to_document(spectrum: Spectrum, n_decimals: int) -> SpectrumDocument:
        return SpectrumDocument(spectrum, n_decimals=n_decimals)
