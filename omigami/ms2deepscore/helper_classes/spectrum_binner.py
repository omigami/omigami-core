from itertools import repeat
from logging import Logger
from multiprocessing import Pool, cpu_count
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
        self,
        spectra: List[Spectrum],
        progress_logger: TaskProgressLogger = None,
        logger: Logger = None,
    ) -> List[BinnedSpectrum]:
        spectra_ids = [spectrum.metadata["spectrum_id"] for spectrum in spectra]

        with Pool(cpu_count()) as pool:
            binned_spectra = pool.starmap(
                fit_transform,
                zip(
                    spectra,
                    repeat(self.spectrum_binner),
                    repeat(progress_logger).repeat(logger),
                ),
            )

        binned_spectra = [
            spectrum.set("spectrum_id", spectra_ids[i])
            for i, spectrum in enumerate(binned_spectra)
        ]
        return binned_spectra


# TODO: These two transform functions don't need to be reimplemented. Remove after debugging.
def fit_transform(
    spectrums: List[Spectrum],
    spectrum_binner: SpectrumBinner,
    progress_logger: TaskProgressLogger = None,
    logger: Logger = None,
):
    if logger:
        logger.info("Collect spectrum peaks...")
    peak_to_position, known_bins = unique_peaks_fixed(
        spectrums,
        spectrum_binner.d_bins,
        spectrum_binner.mz_max,
        spectrum_binner.mz_min,
    )
    if logger:
        logger.info(f"Calculated embedding dimension: {len(known_bins)}.")
    spectrum_binner.peak_to_position = peak_to_position
    spectrum_binner.known_bins = known_bins

    if logger:
        logger.info("Convert spectrums to binned spectrums...")
    return transform(spectrums, spectrum_binner, progress_logger=progress_logger)


def transform(
    input_spectrums: List[Spectrum],
    spectrum_binner: SpectrumBinner,
    progress_logger: TaskProgressLogger = None,
) -> List[BinnedSpectrum]:
    peak_lists, missing_fractions = create_peak_list_fixed(
        input_spectrums,
        spectrum_binner.peak_to_position,
        spectrum_binner.d_bins,
        mz_max=spectrum_binner.mz_max,
        mz_min=spectrum_binner.mz_min,
        peak_scaling=spectrum_binner.peak_scaling,
        progress_bar=False,
    )
    spectrums_binned = []
    for i, peak_list in enumerate(peak_lists):
        assert (
            100 * missing_fractions[i] <= spectrum_binner.allowed_missing_percentage
        ), f"{100*missing_fractions[i]:.2f} of weighted spectrum is unknown to the model."
        spectrum = BinnedSpectrum(
            binned_peaks=create_peak_dict(peak_list),
            metadata={"inchikey": input_spectrums[i].get("inchikey")},
        )
        spectrums_binned.append(spectrum)

        if progress_logger:
            progress_logger.log(i)

    return spectrums_binned
