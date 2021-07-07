from typing import List

from matchms import Spectrum
from ms2deepscore import BinnedSpectrum
from ms2deepscore.models import load_model as ms2deepscore_load_model


class SpectrumBinner:
    def bin_spectra(
        self, spectra: List[Spectrum], model_path: str
    ) -> List[BinnedSpectrum]:
        model = ms2deepscore_load_model(model_path)
        spectra_ids = [spectrum.metadata["spectrum_id"] for spectrum in spectra]
        binned_spectra = model.spectrum_binner.transform(spectra)
        binned_spectra = [
            spectrum.set("spectrum_id", spectra_ids[i])
            for i, spectrum in enumerate(binned_spectra)
        ]
        return binned_spectra
