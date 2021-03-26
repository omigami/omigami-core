from matchms import Spectrum
from spec2vec import SpectrumDocument


class DocumentConverter:
    def __init__(self):
        pass

    @staticmethod
    def convert_to_document(spectrum: Spectrum, n_decimals: int) -> SpectrumDocument:
        return SpectrumDocument(spectrum, n_decimals=n_decimals)
