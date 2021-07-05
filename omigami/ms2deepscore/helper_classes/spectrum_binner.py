from typing import List

from matchms import Spectrum
from ms2deepscore import BinnedSpectrum
from ms2deepscore.models import load_model as ms2deepscore_load_model


class SpectrumBinner:
    def __init__(self, model_path: str):
        self.model = ms2deepscore_load_model(model_path)

    def bin_spectra(self, spectra: List[Spectrum]) -> List[BinnedSpectrum]:
        spectra_ids = [spectrum.metadata["spectrum_id"] for spectrum in spectra]
        binned_spectra = self.model.spectrum_binner.transform(spectra)
        binned_spectra = [
            spectrum.set("spectrum_id", spectra_ids[i])
            for i, spectrum in enumerate(binned_spectra)
        ]
        return binned_spectra
