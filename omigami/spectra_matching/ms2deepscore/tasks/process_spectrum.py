from dataclasses import dataclass
from typing import Set, List

import prefect
from prefect import Task

from omigami.common.progress_logger import TaskProgressLogger
from omigami.config import IonModes
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
    ion_mode: IonModes
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
        self._ion_mode = process_parameters.ion_mode
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

        loaded_cleaned_spectra = []
        for path in cleaned_spectrum_paths:
            loaded_cleaned_spectra += self._fs_gtw.read_from_file(path)

        spectra_to_process_num = len(loaded_cleaned_spectra)
        self.logger.info(f"Cleaning and binning {spectra_to_process_num} spectra")

        progress_logger = TaskProgressLogger(
            self.logger, spectra_to_process_num, 20, "Process Spectra task progress"
        )
        cleaned_spectra = self._processor.process_spectra(
            loaded_cleaned_spectra, process_reference_spectra=True, progress_logger=progress_logger
        )
        binned_spectra = self._spectrum_binner.bin_spectra(cleaned_spectra)

        # saves spectrum binner to filesystem
        self._fs_gtw.serialize_to_file(
            self._spectrum_binner_output_path, self._spectrum_binner.spectrum_binner
        )

        if not binned_spectra or len(binned_spectra) == 0:
            self.logger.info("No new spectra have been processed.")
            return {spectrum.get("spectrum_id") for spectrum in loaded_cleaned_spectra}

        # saves binned spectra to filesystem
        self._fs_gtw.serialize_to_file(self._binned_spectra_output_path, binned_spectra)

        self.logger.info(
            f"Finished processing {len(binned_spectra)} binned spectra. "
            f"Saving into spectrum database."
        )

        return {spectrum.get("spectrum_id") for spectrum in binned_spectra}



