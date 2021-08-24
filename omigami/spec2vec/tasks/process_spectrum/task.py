import os
from dataclasses import dataclass
from typing import Set, List

from prefect import Task

from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways.fs_document_gateway import (
    Spec2VecFSDataGateway,
)
from omigami.spec2vec.gateways.gateway_controller import Spec2VecGatewayController
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)
from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger
from omigami.spec2vec.tasks.process_spectrum.spectrum_processor import (
    SpectrumProcessor,
)
from omigami.utils import merge_prefect_task_configs


@dataclass
class ProcessSpectrumParameters:
    spectrum_dgw: Spec2VecRedisSpectrumDataGateway
    documents_save_directory: str
    ion_mode: str
    n_decimals: int = 2
    overwrite_all_spectra: bool = True


class ProcessSpectrum(Task):
    def __init__(
        self,
        data_gtw: Spec2VecFSDataGateway,
        process_parameters: ProcessSpectrumParameters,
        **kwargs,
    ):
        self._data_gtw = data_gtw
        self._spectrum_dgw = process_parameters.spectrum_dgw
        self._n_decimals = process_parameters.n_decimals
        self._overwrite_all_spectra = process_parameters.overwrite_all_spectra
        self._processor = SpectrumProcessor()
        self._documents_save_directory = process_parameters.documents_save_directory
        self._ion_mode = process_parameters.ion_mode
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
                    f"Saving into spectrum filesystem."
                )

                chunk_count = self._get_chunk_count(self._documents_save_directory)
                spectrum_documents = [doc.document for doc in spectrum_documents]

                document_save_directory = (
                    f"{self._documents_save_directory}/documents{chunk_count}.pickle"
                )

                self.logger.info(f"Saving documents to {document_save_directory}.")

                dgw_controller = Spec2VecGatewayController(self._ion_mode)
                dgw_controller.write_documents(
                    document_save_directory, spectrum_documents
                )

                # TODO: Delete
                self.logger.info(f"Returning in if {document_save_directory}.")
                self.logger.info(
                    f"File exists {self._data_gtw.exists(document_save_directory)}."
                )
                self.logger.info(f"FS: {self._data_gtw.fs}")
                return document_save_directory

        self.logger.info("All spectra have already been processed.")
        # TODO: Delete
        self.logger.info(f"Returning not in if {self._documents_save_directory}.")
        return self._documents_save_directory

    def _get_spectrum_ids_to_add(self, spectrum_ids: List[str]) -> List[str]:
        self.logger.info(
            f"Flag overwrite_all_spectra is set to {self._overwrite_all_spectra}."
        )
        if self._overwrite_all_spectra:
            spectrum_ids_to_add = spectrum_ids
        else:
            spectrum_ids_to_add = self._spectrum_dgw.list_missing_documents(
                spectrum_ids, self._ion_mode
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

        if not self._data_gtw.exists(documents_save_directory):
            self._data_gtw.makedirs(documents_save_directory)

        # TODO: Delete
        self.logger.info(
            f"Files found  {self._data_gtw.listdir(self._documents_save_directory)}."
        )

        return len(self._data_gtw.listdir(self._documents_save_directory))
