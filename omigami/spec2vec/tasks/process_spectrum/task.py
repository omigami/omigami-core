import os
from dataclasses import dataclass
from typing import Set, List

from prefect import Task

from omigami.gateways import RedisSpectrumDataGateway
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways.fs_document_gateway import (
    Spec2VecFSDocumentDataGateway,
)
from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger
from omigami.spec2vec.tasks.process_spectrum.spectrum_processor import (
    SpectrumProcessor,
)
from omigami.utils import merge_prefect_task_configs


@dataclass
class ProcessSpectrumParameters:
    spectrum_dgw: RedisSpectrumDataGateway
    documents_save_directory: str
    n_decimals: int = 2
    overwrite_all_spectra: bool = True


class ProcessSpectrum(Task):
    def __init__(
        self,
        data_gtw: Spec2VecFSDocumentDataGateway,
        process_parameters: ProcessSpectrumParameters,
        **kwargs,
    ):
        self._data_gtw = data_gtw
        self._spectrum_dgw = process_parameters.spectrum_dgw
        self._n_decimals = process_parameters.n_decimals
        self._overwrite_all_spectra = process_parameters.overwrite_all_spectra
        self._processor = SpectrumProcessor()
        self._documents_save_directory = process_parameters.documents_save_directory
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, spectrum_ids: Set[str] = None) -> str:
        self.logger.info(f"Processing {len(spectrum_ids)} spectra")

        spectrum_ids_to_add = self._get_spectrum_ids_to_add(list(spectrum_ids))

        if spectrum_ids_to_add:
            spectrum_documents = self._get_documents(spectrum_ids_to_add)
            if spectrum_documents:
                self.logger.info(
                    f"Finished processing {len(spectrum_documents)}. "
                    f"Saving into spectrum database."
                )

                chunk_count = self._get_chunk_count(self._documents_save_directory)

                document_save_directory = (
                    f"{self._documents_save_directory}/documents{chunk_count}.pkl"
                )

                self._data_gtw.serialize_spectrum_documents(
                    document_save_directory, spectrum_documents
                )

                return document_save_directory

        self.logger.info("All spectra have already been processed.")
        return self._documents_save_directory

    def _get_spectrum_ids_to_add(self, spectrum_ids: List[str]) -> List[str]:
        self.logger.info(
            f"Flag overwrite_all_spectra is set to {self._overwrite_all_spectra}."
        )
        if self._overwrite_all_spectra:
            spectrum_ids_to_add = spectrum_ids
        else:
            spectrum_ids_to_add = self._data_gtw.list_missing_documents(
                spectrum_ids, self._documents_save_directory
            )
            self.logger.info(
                f"{len(spectrum_ids_to_add)} out of {len(spectrum_ids)} spectra are "
                f"new and will be processed. "
            )
        return spectrum_ids_to_add

    def _get_documents(self, spectrum_ids: List[str]) -> List[SpectrumDocumentData]:
        spectra = self._spectrum_dgw.read_spectra(spectrum_ids)
        self.logger.info(
            f"Processing {len(spectra)} spectra and converting into " f"documents."
        )

        progress_logger = TaskProgressLogger(
            self.logger, len(spectra), 20, "Process Spectra task progress"
        )
        return self._processor.create_documents(
            spectra, n_decimals=self._n_decimals, progress_logger=progress_logger
        )

    def _get_chunk_count(self, documents_save_directory) -> int:

        if not os.path.exists(documents_save_directory):
            os.mkdir(documents_save_directory)

        return len(os.listdir(self._documents_save_directory))
