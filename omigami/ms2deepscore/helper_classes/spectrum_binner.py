from typing import List

from matchms import Spectrum
from ms2deepscore import BinnedSpectrum, SpectrumBinner
from ms2deepscore.spectrum_binning_fixed import (
    unique_peaks_fixed,
    create_peak_list_fixed,
)
from ms2deepscore.utils import create_peak_dict
from tqdm import tqdm


class MS2DeepScoreSpectrumBinner:
    def __init__(self, n_bins: int = 10000):
        self.spectrum_binner = SpectrumBinner(number_of_bins=n_bins)

    def bin_spectra(self, spectra: List[Spectrum]) -> List[BinnedSpectrum]:
        # binned_spectra = self.spectrum_binner.fit_transform(spectra)

        print("Collect spectrum peaks...")
        peak_to_position, known_bins = unique_peaks_fixed(
            spectra,
            self.spectrum_binner.d_bins,
            self.spectrum_binner.mz_max,
            self.spectrum_binner.mz_min,
        )
        print(f"Calculated embedding dimension: {len(known_bins)}.")
        self.spectrum_binner.peak_to_position = peak_to_position
        self.spectrum_binner.known_bins = known_bins

        print("Convert spectrums to binned spectrums...")

        peak_lists, missing_fractions = create_peak_list_fixed(
            spectra,
            self.spectrum_binner.peak_to_position,
            self.spectrum_binner.d_bins,
            mz_max=self.spectrum_binner.mz_max,
            mz_min=self.spectrum_binner.mz_min,
            peak_scaling=self.spectrum_binner.peak_scaling,
            progress_bar=True,
        )
        binned_spectra = []
        for i, peak_list in enumerate(
            tqdm(peak_lists, desc="Create BinnedSpectrum instances", disable=(not True))
        ):
            assert (
                100 * missing_fractions[i]
                <= self.spectrum_binner.allowed_missing_percentage
            ), f"{100*missing_fractions[i]:.2f} of weighted spectrum is unknown to the model. GUILY={spectra[i].get('spectrum_id')}"
            spectrum = BinnedSpectrum(
                binned_peaks=create_peak_dict(peak_list),
                metadata={"inchikey": spectra[i].get("inchikey")},
            )
            binned_spectra.append(spectrum)

        for binned_spectrum, spectrum in zip(binned_spectra, spectra):
            binned_spectrum.set("spectrum_id", spectrum.get("spectrum_id"))
            binned_spectrum.set("inchi", spectrum.get("inchi"))
        return binned_spectra
