from dataclasses import dataclass
from pathlib import Path

from typing import Set, List

import prefect
from prefect import Task

from omigami.common.progress_logger import TaskProgressLogger
from omigami.spectra_matching.ms2deepscore.helper_classes.spectrum_binner import (
    MS2DeepScoreSpectrumBinner,
)
from omigami.spectra_matching.ms2deepscore.helper_classes.spectrum_processor import (
    SpectrumProcessor,
)

from omigami.spectra_matching.storage import DataGateway
from omigami.utils import merge_prefect_task_configs


@dataclass
class ProcessSpectrumParameters:
    spectrum_binner_output_path: str
    binned_spectra_output_path: str
    n_bins: int = 10000


class ProcessSpectrum(Task):
    def __init__(
        self,
        fs_gtw: DataGateway,
        process_parameters: ProcessSpectrumParameters,
        **kwargs,
    ):
        self._fs_gtw = fs_gtw
        self._spectrum_binner_output_path = (
            process_parameters.spectrum_binner_output_path
        )
        self._binned_spectra_output_path = (
            process_parameters.binned_spectra_output_path
        )
        self._processor = SpectrumProcessor()
        self._spectrum_binner = MS2DeepScoreSpectrumBinner(process_parameters.n_bins)
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config, trigger=prefect.triggers.all_successful)

    def run(self, cleaned_spectrum_paths: List[str]) -> Set[str]:
        """
        Prefect task to clean spectra and create binned spectra from cleaned spectra.
        Binned spectra are saved to REDIS DB and filesystem.

        Parameters
        ----------
        cleaned_spectrum_paths: List[str]
            paths to the cleaned spectra.

        Returns
        -------
        Set of spectrum_ids
        """

        self.logger.info(f"Loading cleaned spectra from directory {Path(cleaned_spectrum_paths[0]).parent}.")
        cleaned_spectra = []
        for path in cleaned_spectrum_paths:
            cleaned_spectra += self._fs_gtw.read_from_file(path)

        cleaned_spectra_size = len(cleaned_spectra)
        self.logger.info(f"Cleaning and binning {cleaned_spectra_size} spectra")

        progress_logger = TaskProgressLogger(
            self.logger, cleaned_spectra_size, 20, "Process Spectra task progress"
        )
        cleaned_spectra = self._processor.process_spectra(
            cleaned_spectra, process_reference_spectra=True, progress_logger=progress_logger
        )
        binned_spectra = self._spectrum_binner.bin_spectra(cleaned_spectra)

        # saves spectrum binner to filesystem
        self._fs_gtw.serialize_to_file(
            self._spectrum_binner_output_path, self._spectrum_binner.spectrum_binner
        )

        if not binned_spectra or len(binned_spectra) == 0:
            self.logger.info("No new spectra have been processed.")
            return {spectrum.get("spectrum_id") for spectrum in cleaned_spectra}

        # saves binned spectra to filesystem
        self._fs_gtw.serialize_to_file(self._binned_spectra_output_path, binned_spectra)

        self.logger.info(
            f"Finished processing {len(binned_spectra)} binned spectra. "
            f"Saving into spectrum database."
        )

        return {spectrum.get("spectrum_id") for spectrum in binned_spectra}



