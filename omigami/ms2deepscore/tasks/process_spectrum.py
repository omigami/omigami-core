from copy import deepcopy
from dataclasses import dataclass
from typing import Set
from prefect import Task
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.helper_classes.spectrum_binner import (
    MS2DeepScoreSpectrumBinner,
)
from omigami.ms2deepscore.helper_classes.spectrum_processor import (
    SpectrumProcessor,
)
from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger
from omigami.utils import merge_prefect_task_configs


@dataclass
class ProcessSpectrumParameters:
    spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway
    overwrite_all: bool = True
    is_minimal_flow: bool = False


class ProcessSpectrum(Task):
    def __init__(
        self,
        process_parameters: ProcessSpectrumParameters,
        **kwargs,
    ):
        self._spectrum_dgw = process_parameters.spectrum_dgw
        self._overwrite_all = process_parameters.overwrite_all
        self._processor = SpectrumProcessor(process_parameters.is_minimal_flow)
        self._spectrum_binner = MS2DeepScoreSpectrumBinner()
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, spectrum_ids: Set[str] = None) -> Set[str]:
        self.logger.info(f"Processing {len(spectrum_ids)} spectra")
        all_spectrum_ids = deepcopy(spectrum_ids)

        self.logger.info(f"Flag skip_if_exists is set to {self._overwrite_all}.")
        if self._overwrite_all:
            missing_ids = self._spectrum_dgw.list_missing_binned_spectra(
                list(spectrum_ids)
            )
            if not missing_ids:
                self.logger.info("All spectra have already been processed.")
                return set(all_spectrum_ids)

            self.logger.info(
                f"{len(missing_ids)} out of {len(spectrum_ids)} spectra are new and "
                f"will be processed."
            )
            spectrum_ids = missing_ids

        spectra = self._spectrum_dgw.read_spectra(spectrum_ids)

        self.logger.info(f"Processing {len(spectra)} spectra")

        self.logger.info(f"Cleaning spectra and binning them")
        progress_logger = TaskProgressLogger(
            self.logger, len(spectra), 20, "Process Spectra task progress"
        )
        cleaned_spectra = self._processor.process_spectra(
            spectra, process_reference_spectra=True, progress_logger=progress_logger
        )
        binned_spectra = self._spectrum_binner.bin_spectra(cleaned_spectra)

        if self._overwrite_all and not binned_spectra:
            self.logger.info("No new spectra have been processed.")
            return set(all_spectrum_ids)

        self.logger.info(
            f"Finished processing {len(binned_spectra)} binned spectra. "
            f"Saving into spectrum database."
        )
        self._spectrum_dgw.write_binned_spectra(binned_spectra)

        return {sp.metadata["spectrum_id"] for sp in binned_spectra}
