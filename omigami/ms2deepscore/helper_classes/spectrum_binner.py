from typing import List

from matchms import Spectrum
from ms2deepscore import BinnedSpectrum, SpectrumBinner
from ms2deepscore.spectrum_binning_fixed import (
    unique_peaks_fixed,
    create_peak_list_fixed,
)
from ms2deepscore.utils import create_peak_dict

from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger


class MS2DeepScoreSpectrumBinner:
    def __init__(self):
        self.spectrum_binner = SpectrumBinner(number_of_bins=10000)

    def bin_spectra(
        self, spectra: List[Spectrum], progress_logger: TaskProgressLogger = None
    ) -> List[BinnedSpectrum]:
        spectra_ids = [spectrum.metadata["spectrum_id"] for spectrum in spectra]
        binned_spectra = self.fit_transform(spectra, progress_logger=progress_logger)
        binned_spectra = [
            spectrum.set("spectrum_id", spectra_ids[i])
            for i, spectrum in enumerate(binned_spectra)
        ]
        return binned_spectra

    def fit_transform(
        self, spectrums: List[Spectrum], progress_logger: TaskProgressLogger = None
    ):
        print("Collect spectrum peaks...")
        peak_to_position, known_bins = unique_peaks_fixed(
            spectrums,
            self.spectrum_binner.d_bins,
            self.spectrum_binner.mz_max,
            self.spectrum_binner.mz_min,
        )
        print(f"Calculated embedding dimension: {len(known_bins)}.")
        self.spectrum_binner.peak_to_position = peak_to_position
        self.spectrum_binner.known_bins = known_bins

        print("Convert spectrums to binned spectrums...")
        return self.transform(spectrums, progress_logger=progress_logger)

    def transform(
        self,
        input_spectrums: List[Spectrum],
        progress_logger: TaskProgressLogger = None,
    ) -> List[BinnedSpectrum]:
        peak_lists, missing_fractions = create_peak_list_fixed(
            input_spectrums,
            self.spectrum_binner.peak_to_position,
            self.spectrum_binner.d_bins,
            mz_max=self.spectrum_binner.mz_max,
            mz_min=self.spectrum_binner.mz_min,
            peak_scaling=self.spectrum_binner.peak_scaling,
            progress_bar=False,
        )
        spectrums_binned = []
        for i, peak_list in enumerate(peak_lists):
            assert (
                100 * missing_fractions[i]
                <= self.spectrum_binner.allowed_missing_percentage
            ), f"{100*missing_fractions[i]:.2f} of weighted spectrum is unknown to the model."
            spectrum = BinnedSpectrum(
                binned_peaks=create_peak_dict(peak_list),
                metadata={"inchikey": input_spectrums[i].get("inchikey")},
            )
            spectrums_binned.append(spectrum)

            if progress_logger:
                progress_logger.log(i)

        return spectrums_binned
