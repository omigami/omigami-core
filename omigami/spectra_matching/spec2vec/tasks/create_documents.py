from dataclasses import dataclass
from typing import List

from drfs import DRPath
from matchms import Spectrum
from matchms.filtering import normalize_intensities
from prefect import Task

from omigami.common.progress_logger import (
    TaskProgressLogger,
)
from omigami.spectra_matching.spec2vec.entities.spectrum_document import (
    SpectrumDocumentData,
)
from omigami.spectra_matching.storage import FSDataGateway
from omigami.utils import merge_prefect_task_configs


@dataclass
class CreateDocumentsParameters:
    output_directory: str
    ion_mode: str
    n_decimals: int = 2


class CreateDocuments(Task):
    def __init__(
        self,
        fs_dgw: FSDataGateway,
        parameters: CreateDocumentsParameters,
        **kwargs,
    ):
        self._fs_dgw = fs_dgw
        self._n_decimals = parameters.n_decimals
        self._output_directory = parameters.output_directory
        self._ion_mode = parameters.ion_mode
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config, checkpoint=True)

    def run(self, cleaned_spectra_path: str = None) -> str:
        """
        Prefect task to create spectrum documents from input spectra. Spectrum documents
        are saved to filesystem.

        Parameters
        ----------
        cleaned_spectra_path:
            Path containing the spectra to be processed

        Returns
        -------
        Directory which documents are saved

        """
        self.logger.info(f"Loading spectra from path {cleaned_spectra_path}.")
        spectra: List[Spectrum] = self._fs_dgw.read_from_file(cleaned_spectra_path)
        document_output_path = (
            f"{self._output_directory}/{DRPath(cleaned_spectra_path).name}"
        )

        if DRPath(document_output_path).exists() and self.checkpoint:
            self.logger.info(f"Using cached existing file on {document_output_path}")
            return document_output_path

        self.logger.info(
            f"Processing {len(spectra)} spectra and converting into " f"documents."
        )
        documents = self._create_documents(spectra)

        self.logger.info(
            f"Finished processing {len(documents)}. Saving into spectrum filesystem."
        )

        # this is weird. if we only use doc.documents why do we need the rest?
        spectrum_documents = [doc.document for doc in documents]
        self.logger.info(f"Saving documents to {document_output_path}.")
        self._fs_dgw.serialize_to_file(document_output_path, spectrum_documents)

        return document_output_path

    def _create_documents(
        self, spectra: List[Spectrum], min_peaks: int = 0
    ) -> List[SpectrumDocumentData]:
        progress_logger = TaskProgressLogger(
            self.logger, len(spectra), 20, "Process Spectra task progress"
        )
        documents = []
        for i, spectrum in enumerate(spectra):
            if spectrum is not None and len(spectrum.peaks.mz) > min_peaks:
                processed_spectrum = normalize_intensities(spectrum)
                document = SpectrumDocumentData(processed_spectrum, self._n_decimals)

                if document.document:
                    documents.append(document)

                if progress_logger:
                    progress_logger.log(i)

        return documents
