from dataclasses import dataclass
from typing import Set, List, Tuple

import prefect
from ms2deepscore import SpectrumBinner
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
    overwrite_all_spectra: bool = True
    is_pretrained_flow: bool = False


class ProcessSpectrum(Task):
    def __init__(
        self,
        process_parameters: ProcessSpectrumParameters,
        **kwargs,
    ):
        self._spectrum_dgw = process_parameters.spectrum_dgw
        self._overwrite_all_spectra = process_parameters.overwrite_all_spectra
        self._processor = SpectrumProcessor(process_parameters.is_pretrained_flow)
        self._spectrum_binner = MS2DeepScoreSpectrumBinner()
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config, trigger=prefect.triggers.all_successful)

    def run(
        self, spectrum_ids_chunks: List[Set[str]] = None
    ) -> Tuple[Set[str], SpectrumBinner]:
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
            self._spectrum_dgw.write_binned_spectra(binned_spectra)
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
        return self._spectrum_binner.bin_spectra(cleaned_spectra)
