from dataclasses import dataclass
from typing import Set, List

import prefect
from prefect import Task

from omigami.config import IonModes
from omigami.spectra_matching.gateways import DataGateway
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
    spectrum_binner_output_path: str
    ion_mode: IonModes
    overwrite_all_spectra: bool = True
    is_pretrained_flow: bool = False
    n_bins: int = 10000


class ProcessSpectrum(Task):
    def __init__(
        self,
        fs_gtw: DataGateway,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        process_parameters: ProcessSpectrumParameters,
        **kwargs,
    ):
        self._fs_gtw = fs_gtw
        self._spectrum_dgw = spectrum_dgw
        self._overwrite_all_spectra = process_parameters.overwrite_all_spectra
        self._spectrum_binner_output_path = (
            process_parameters.spectrum_binner_output_path
        )
        self._ion_mode = process_parameters.ion_mode
        self._processor = SpectrumProcessor(process_parameters.is_pretrained_flow)
        self._spectrum_binner = MS2DeepScoreSpectrumBinner(process_parameters.n_bins)
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config, trigger=prefect.triggers.all_successful)

    def run(self, spectrum_ids_chunks: List[Set[str]] = None) -> Set[str]:
        """
        Prefect task to clean spectra and create binned spectra from cleaned spectra.
        Binned spectra are saved to REDIS DB and filesystem.

        Parameters
        ----------
        spectrum_ids_chunks: List[Set[str]]
            spectrum_ids defined in the set of chunks. If it is not passed, then method
            cleans and creates binned spectra for the existing ones in DB.

        Returns
        -------
        Set of spectrum_ids

        """
        if spectrum_ids_chunks:
            spectrum_ids = [item for elem in spectrum_ids_chunks for item in elem]
        else:
            spectrum_ids = self._spectrum_dgw.list_spectrum_ids()
        self.logger.info(f"Processing {len(spectrum_ids)} spectra")

        binned_spectra = self._get_binned_spectra(spectrum_ids)
        if binned_spectra:
            self.logger.info(
                f"Finished processing {len(binned_spectra)} binned spectra. "
                f"Saving into spectrum database."
            )
            self._spectrum_dgw.write_binned_spectra(
                binned_spectra, self._ion_mode, self.logger
            )
            return {sp.get("spectrum_id") for sp in binned_spectra}

        self.logger.info("No new spectra have been processed.")
        return set(spectrum_ids)

    def _get_binned_spectra(self, spectrum_ids: List[str]):
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

        self._fs_gtw.serialize_to_file(
            self._spectrum_binner_output_path, self._spectrum_binner.spectrum_binner
        )

        return binned_spectra
