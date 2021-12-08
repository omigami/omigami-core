from typing import List

from matchms import Spectrum
from matchms.filtering import normalize_intensities

from omigami.common.progress_logger import (
    TaskProgressLogger,
)
from omigami.spectra_matching.spec2vec.entities.spectrum_document import (
    SpectrumDocumentData,
)
from omigami.spectra_matching.spectrum_cleaner import SpectrumCleaner


class SpectrumProcessor(SpectrumCleaner):
    def create_documents(
        self,
        spectra: List[Spectrum],
        min_peaks: int = 0,
        n_decimals: int = 2,
        progress_logger: TaskProgressLogger = None,
    ) -> List[SpectrumDocumentData]:
        # TODO: there is something wrong with this code compared to what we had before.
        # need to investigate.
        documents = []
        for i, spectrum in enumerate(spectra):
            if spectrum is not None and len(spectrum.peaks.mz) > min_peaks:
                processed_spectrum = normalize_intensities(spectrum)
                document = SpectrumDocumentData(processed_spectrum, n_decimals)

                if document.document:
                    documents.append(document)

                if progress_logger:
                    progress_logger.log(i)

        return documents
