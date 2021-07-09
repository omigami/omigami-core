import multiprocessing
from typing import List
import numpy as np

from joblib import Parallel, delayed
from matchms import Spectrum
from ms2deepscore import BinnedSpectrum, SpectrumBinner
from ms2deepscore.spectrum_binning_fixed import (
    unique_peaks_fixed,
    create_peak_list_fixed,
    bin_number_array_fixed,
)
from ms2deepscore.utils import create_peak_dict
from prefect.utilities import logging
from tqdm import tqdm

from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger

NUM_CORES = multiprocessing.cpu_count() + 8


class MS2DeepScoreSpectrumBinner:
    def __init__(self, n_bins: int = 10000):
        self.spectrum_binner = SpectrumBinner(number_of_bins=n_bins)
        self.logger = logging.get_logger("MS2DeepScoreSpectrumBinner")

    def bin_spectra(
        self, spectra: List[Spectrum], progress_logger: TaskProgressLogger = None
    ) -> List[BinnedSpectrum]:
        self.logger.info(f"binning {len(spectra)} spectra")
        binned_spectra = self.fit_transform(spectra, progress_logger)

        # add metadata to binned spectra
        binned_spectra = [
            binned_spectrum.set("spectrum_id", raw_spectrum.metadata["spectrum_id"])
            for binned_spectrum, raw_spectrum in zip(binned_spectra, spectra)
        ]

        return binned_spectra

    def fit_transform(
        self, spectrums: List[Spectrum], progress_logger: TaskProgressLogger = None
    ):
        self.logger.info("Collect spectrum peaks...")
        peak_to_position, known_bins = unique_peaks_fixed(
            spectrums,
            self.spectrum_binner.d_bins,
            self.spectrum_binner.mz_max,
            self.spectrum_binner.mz_min,
        )
        self.logger.info(f"Calculated embedding dimension: {len(known_bins)}.")
        self.spectrum_binner.peak_to_position = peak_to_position
        self.spectrum_binner.known_bins = known_bins

        self.logger.info("Convert spectrums to binned spectrums...")
        return self.transform(spectrums, progress_logger=progress_logger)

    def transform(
        self,
        input_spectrums: List[Spectrum],
        progress_logger: TaskProgressLogger = None,
    ) -> List[BinnedSpectrum]:

        self.logger.info("Create peak list fixed...")
        peak_lists, missing_fractions = create_peak_list_fixed(
            input_spectrums,
            self.spectrum_binner.peak_to_position,
            self.spectrum_binner.d_bins,
            mz_max=self.spectrum_binner.mz_max,
            mz_min=self.spectrum_binner.mz_min,
            peak_scaling=self.spectrum_binner.peak_scaling,
            progress_bar=False,
        )

        self.logger.info(f"Binning {len(input_spectrums)} using {NUM_CORES} cores")
        spectrums_binned = Parallel(n_jobs=NUM_CORES, verbose=1000)(
            delayed(self._transform)(spectrum, peak_list, missing_fraction)
            for spectrum, peak_list, missing_fraction in tqdm(
                zip(input_spectrums, peak_lists, missing_fractions)
            )
        )
        spectrums_binned = [
            spectrum for spectrum in spectrums_binned if spectrum is not None
        ]

        return spectrums_binned

    def _transform(
        self, input_spectrum: Spectrum, peak_list: list, missing_fraction: float
    ):
        assert (
            100 * missing_fraction <= self.spectrum_binner.allowed_missing_percentage
        ), f"{100 * missing_fraction:.2f} of weighted spectrum is unknown to the model."

        binned_spectrum = BinnedSpectrum(
            binned_peaks=create_peak_dict(peak_list),
            metadata={"inchikey": input_spectrum.get("inchikey")},
        )
        return binned_spectrum
