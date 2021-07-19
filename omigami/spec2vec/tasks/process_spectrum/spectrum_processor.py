from typing import Dict, List

from matchms.importing.load_from_json import as_spectrum

from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger
from omigami.spectrum_cleaner import SpectrumCleaner


class SpectrumProcessor(SpectrumCleaner):
    def create_documents(
        self,
        spectrum_dicts: List[Dict],
        min_peaks: int = 0,
        n_decimals: int = 2,
        progress_logger: TaskProgressLogger = None,
    ) -> List[SpectrumDocumentData]:
        # TODO: there is something wrong with this code compared to what we had before.
        # need to investigate.
        documents = []
        for i, spectrum_dict in enumerate(spectrum_dicts):
            spectrum = as_spectrum(spectrum_dict)

            if spectrum is not None and len(spectrum.peaks.mz) > min_peaks:
                processed_spectrum = self._convert_metadata(
                    self._harmonize_spectrum(self._apply_filters(spectrum))
                )
                doc = SpectrumDocumentData(processed_spectrum, n_decimals)
                # cz: I'm not sure why this check is needed. looks like some bad implementation
                # somewhere. We should try to investigate this if we have the time

                if doc.spectrum and doc.document:
                    documents.append(doc)

                if progress_logger:
                    progress_logger.log(i)

        return documents
