from dataclasses import dataclass
from typing import Set

from prefect import Task

from omigami.gateways.data_gateway import SpectrumDataGateway
from omigami.gateways.redis_spectrum_data_gateway import REDIS_DB
from omigami.ms2deepscore.helper_classes.spectrum_binner import SpectrumBinner
from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger
from omigami.ms2deepscore.helper_classes.spectrum_processor import (
    SpectrumProcessor,
)
from omigami.utils import merge_prefect_task_configs


@dataclass
class ProcessSpectrumParameters:
    spectrum_dgw: SpectrumDataGateway
    model_path: str
    skip_if_exists: bool = True


class ProcessSpectrum(Task):
    def __init__(
        self,
        process_parameters: ProcessSpectrumParameters,
        **kwargs,
    ):
        self._spectrum_dgw = process_parameters.spectrum_dgw
        self._skip_if_exists = process_parameters.skip_if_exists
        self._model_path = process_parameters.model_path
        self._processor = SpectrumProcessor()
        self._spectrum_binner = SpectrumBinner()
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self) -> Set[str]:
        self.logger.info(f"Using Redis DB {REDIS_DB}")
        spectra_ids = self._spectrum_dgw.list_spectrum_ids()

        self.logger.info(f"Flag skip_if_exists is set to {self._skip_if_exists}.")
        if self._skip_if_exists:
            new_spectrum_ids = self._spectrum_dgw.list_missing_binned_spectra(
                spectra_ids
            )
            if not new_spectrum_ids:
                self.logger.info("All spectra have already been processed.")
                return set(spectra_ids)

            self.logger.info(
                f"{len(new_spectrum_ids)} out of {len(spectra_ids)} spectra are new and will "
                f"be processed."
            )
            spectra_ids = new_spectrum_ids

        spectra = self._spectrum_dgw.read_spectra(spectra_ids)
        self.logger.info(f"Processing {len(spectra)} spectra")

        self.logger.info(f"Cleaning spectra and binning them")
        progress_logger = TaskProgressLogger(
            self.logger, len(spectra), 20, "Process Spectra task progress"
        )
        cleaned_spectra = self._processor.process_spectra(
            list(spectra.values()), progress_logger=progress_logger
        )
        binned_spectra = self._spectrum_binner.bin_spectra(
            cleaned_spectra, self._model_path
        )

        if self._skip_if_exists and not binned_spectra:
            self.logger.info("No new spectra have been processed.")
            return set(spectra_ids)

        self.logger.info(
            f"Finished processing {len(binned_spectra)} binned spectra. "
            f"Saving into spectrum database."
        )
        self._spectrum_dgw.write_binned_spectra(binned_spectra)

        return {sp.metadata["spectrum_id"] for sp in binned_spectra}
