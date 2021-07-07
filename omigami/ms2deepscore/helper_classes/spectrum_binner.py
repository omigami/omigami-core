from typing import List

from matchms import Spectrum
from ms2deepscore import BinnedSpectrum, SpectrumBinner


class MS2DeepScoreSpectrumBinner:
    def bin_spectra(self, spectra: List[Spectrum]) -> List[BinnedSpectrum]:
        spectrum_binner = SpectrumBinner(number_of_bins=10000)
        spectra_ids = [spectrum.metadata["spectrum_id"] for spectrum in spectra]
        binned_spectra = spectrum_binner.transform(spectra)
        binned_spectra = [
            spectrum.set("spectrum_id", spectra_ids[i])
            for i, spectrum in enumerate(binned_spectra)
        ]
        return binned_spectra
