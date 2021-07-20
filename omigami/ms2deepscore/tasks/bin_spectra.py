from typing import List, Tuple

from matchms import Spectrum
from ms2deepscore import SpectrumBinner, BinnedSpectrum
from prefect import Task

from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.utils import merge_prefect_task_configs


class BinSpectra(Task):
    def __init__(
        self,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        number_of_bins: int = 10000,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._n_bins = number_of_bins
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, spectra: List[Spectrum]) -> SpectrumBinner:
        spectrum_binner, binned_spectra = self._bin_spectra(spectra)

        self.logger.info(
            f"Finished processing {len(binned_spectra)} binned spectra. "
            f"Saving into spectrum database."
        )
        self._spectrum_dgw.write_binned_spectra(binned_spectra)

        return spectrum_binner

    def _bin_spectra(
        self,
        spectra: List[Spectrum],
    ) -> Tuple[SpectrumBinner, List[BinnedSpectrum]]:
        spectrum_binner = SpectrumBinner(number_of_bins=self._n_bins)
        binned_spectra = spectrum_binner.fit_transform(spectra)

        for binned_spectrum, spectrum in zip(binned_spectra, spectra):
            binned_spectrum.set("spectrum_id", spectrum.get("spectrum_id"))
            binned_spectrum.set("inchi", spectrum.get("inchi"))

        return spectrum_binner, binned_spectra
