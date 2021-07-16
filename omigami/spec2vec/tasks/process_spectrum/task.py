from copy import deepcopy
from dataclasses import dataclass
from typing import Set
from prefect import Task
from omigami.gateways.data_gateway import InputDataGateway
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
    n_decimals: int = 2
    skip_if_exists: bool = True


class ProcessSpectrum(Task):
    def __init__(
        self,
        input_dgw: InputDataGateway,
        process_parameters: ProcessSpectrumParameters,
        **kwargs,
    ):
        self._input_dgw = input_dgw
        self._spectrum_dgw = process_parameters.spectrum_dgw
        self._n_decimals = process_parameters.n_decimals
        self._skip_if_exists = process_parameters.skip_if_exists
        self._processor = SpectrumProcessor()
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, spectrum_ids: Set[str] = None) -> Set[str]:
        self.logger.info(f"Processing {len(spectrum_ids)} spectra")
        existing_ids = deepcopy(spectrum_ids)
        self.logger.info(f"Flag skip_if_exists is set to {self._skip_if_exists}.")
        if self._skip_if_exists:
            missing_ids = self._spectrum_dgw.list_missing_documents(list(spectrum_ids))
            if not missing_ids:
                self.logger.info("All spectra have already been processed.")
                return set(missing_ids)

            self.logger.info(
                f"{len(missing_ids)} out of {len(spectrum_ids)} spectra are new and will "
                f"be processed."
            )
            spectrum_ids = missing_ids

        spectra = self._spectrum_dgw.read_spectra(spectrum_ids)
        self.logger.info(f"Processing spectra and converting into documents.")
        progress_logger = TaskProgressLogger(
            self.logger, len(spectra), 20, "Process Spectra task progress"
        )
        spectrum_documents = self._processor.create_documents(
            spectra, n_decimals=self._n_decimals, progress_logger=progress_logger
        )

        if self._skip_if_exists and not spectrum_documents:
            self.logger.info("No new documents have been processed.")
            return existing_ids

        self.logger.info(
            f"Finished processing {len(spectrum_documents)}. "
            f"Saving into spectrum database."
        )
        self._spectrum_dgw.write_spectrum_documents(spectrum_documents)

        return {sp.spectrum_id for sp in spectrum_documents}
