from typing import List

from matchms import Spectrum
from ms2deepscore import BinnedSpectrum
from ms2deepscore.spectrum_binning_fixed import (
    unique_peaks_fixed,
    create_peak_list_fixed,
    set_d_bins_fixed,
)
from ms2deepscore.utils import create_peak_dict
from tqdm import tqdm


class MS2DeepScoreSpectrumBinner:
    def __init__(self):
        self.mz_min = 10
        self.mz_max = 1000
        self.allowed_missing_percentage = 0
        self.d_bins = set_d_bins_fixed(10000, mz_min=10, mz_max=1000)
        self.peak_scaling = 0.5
        self.peak_to_position = None
        self.known_bins = None

    def bin_spectra(self, spectra: List[Spectrum], logger=None) -> List[BinnedSpectrum]:
        binned_spectra = self.fit_transform(spectra, logger)

        for binned_spectrum, spectrum in zip(binned_spectra, spectra):
            binned_spectrum.set("spectrum_id", spectrum.get("spectrum_id"))
            binned_spectrum.set("inchi", spectrum.get("inchi"))
        return binned_spectra

    def fit_transform(self, spectrums: List[Spectrum], progress_bar=True, logger=None):
        """Transforms the input *spectrums* into binned spectrums as needed for
        MS2DeepScore.

        Includes creating a 'vocabulary' of bins that have peaks in spectrums,
        which is stored in SpectrumBinner.known_bins.
        Creates binned spectrums from input spectrums and returns them.

        Parameters
        ----------
        spectrums
            List of spectrums.
        progress_bar
            Show progress bar if set to True. Default is True.
        """
        print("Collect spectrum peaks...")

        peak_to_position, known_bins = unique_peaks_fixed(
            spectrums, self.d_bins, self.mz_max, self.mz_min
        )
        print(f"Calculated embedding dimension: {len(known_bins)}.")
        self.peak_to_position = peak_to_position
        self.known_bins = known_bins

        print("Convert spectrums to binned spectrums...")
        return self.transform(spectrums, progress_bar, logger)

    def transform(
        self, input_spectrums: List[Spectrum], progress_bar=True, logger=None
    ) -> List[BinnedSpectrum]:
        """Create binned spectrums from input spectrums.

        Parameters
        ----------
        input_spectrums
            List of spectrums.
        progress_bar
            Show progress bar if set to True. Default is True.

        Returns:
            List of binned spectrums created from input_spectrums.
        """
        peak_lists, missing_fractions = create_peak_list_fixed(
            input_spectrums,
            self.peak_to_position,
            self.d_bins,
            mz_max=self.mz_max,
            mz_min=self.mz_min,
            peak_scaling=self.peak_scaling,
            progress_bar=progress_bar,
        )
        spectrums_binned = []
        for i, peak_list in enumerate(
            tqdm(
                peak_lists,
                desc="Create BinnedSpectrum instances",
                disable=(not progress_bar),
            )
        ):
            if not (100 * missing_fractions[i] <= self.allowed_missing_percentage):
                if logger:
                    logger.info(
                        f"{100*missing_fractions[i]:.2f} of weighted spectrum is unknown to the model."
                    )
                    logger.info(f"Missing fractions: {missing_fractions[i]}")
                    logger.info(f"Peak list: {peak_list}")
                    logger.info(f"Input spectrum: {input_spectrums[i]}")
                raise AssertionError

            spectrum = BinnedSpectrum(
                binned_peaks=create_peak_dict(peak_list),
                metadata={"inchikey": input_spectrums[i].get("inchikey")},
            )
            spectrums_binned.append(spectrum)
        return spectrums_binned
