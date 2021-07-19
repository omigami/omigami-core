from typing import List
import numpy as np
from matchms import Spectrum
from matchms.typing import SpectrumType
from ms2deepscore import BinnedSpectrum, SpectrumBinner
from ms2deepscore.spectrum_binning_fixed import (
    unique_peaks_fixed,
    bin_number_array_fixed,
)
from ms2deepscore.typing import BinnedSpectrumType
from ms2deepscore.utils import create_peak_dict

from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger


class MS2DeepScoreSpectrumBinner:
    def __init__(self):
        self.spectrum_binner = SpectrumBinner(number_of_bins=10000)

    def bin_spectra(
        self,
        spectra: List[Spectrum],
        progress_logger: TaskProgressLogger = None,
    ) -> List[BinnedSpectrum]:

        binned_spectra = self.fit_transform(spectra, progress_logger)

        for binned_spectrum, spectrum in zip(binned_spectra, spectra):
            binned_spectrum.set("spectrum_id", spectrum.get("spectrum_id"))
            binned_spectrum.set("inchi", spectrum.get("inchi"))
        return binned_spectra

    def fit_transform(
        self,
        spectrums: List[SpectrumType],
        progress_logger: TaskProgressLogger = None,
    ):
        progress_logger.logger.info("Collect spectrum peaks...")
        peak_to_position, known_bins = unique_peaks_fixed(
            spectrums,
            self.spectrum_binner.d_bins,
            self.spectrum_binner.mz_max,
            self.spectrum_binner.mz_min,
        )
        progress_logger.logger.info(
            f"Calculated embedding dimension: {len(known_bins)}."
        )
        self.spectrum_binner.peak_to_position = peak_to_position
        self.spectrum_binner.known_bins = known_bins

        progress_logger.logger.info("Convert spectrums to binned spectrums...")
        return self.transform(spectrums, progress_logger)

    def transform(
        self,
        input_spectrums: List[SpectrumType],
        progress_logger: TaskProgressLogger = None,
    ) -> List[BinnedSpectrumType]:
        progress_logger.logger.info("Start binning spectra.")

        peak_lists, missing_fractions = self._create_peak_list_fixed(
            input_spectrums,
            self.spectrum_binner.peak_to_position,
            self.spectrum_binner.d_bins,
            mz_max=self.spectrum_binner.mz_max,
            mz_min=self.spectrum_binner.mz_min,
            peak_scaling=self.spectrum_binner.peak_scaling,
            progress_logger=progress_logger,
        )
        progress_logger.logger.info(
            f"len comparision {len(peak_lists)} - {len(missing_fractions)}"
        )
        progress_logger.logger.info("Finished binning spectra.")

        spectrums_binned = []
        for ix, peak_list in peak_lists:

            miss_rate = 100 * missing_fractions[ix]
            assert (
                miss_rate <= self.spectrum_binner.allowed_missing_percentage,
                f"{miss_rate:.2f} of weighted spectrum is unknown to the model.",
            )

            spectrum = BinnedSpectrum(
                binned_peaks=create_peak_dict(peak_list),
                metadata={"inchikey": input_spectrums[ix].get("inchikey")},
            )
            spectrums_binned.append(spectrum)

            if progress_logger:
                progress_logger.log(ix, "Creating BinnedSpectrum object")

        return spectrums_binned

    @staticmethod
    def _create_peak_list_fixed(
        spectrums,
        peaks_vocab,
        d_bins,
        mz_max=1000.0,
        mz_min=10.0,
        peak_scaling=0.5,
        progress_logger: TaskProgressLogger = None,
    ):
        peak_lists = []
        missing_fractions = []

        for i, spectrum in enumerate(spectrums):
            if progress_logger:
                progress_logger.log(i, "Binning Spectra")

            doc = bin_number_array_fixed(
                spectrum.peaks.mz, d_bins, mz_max=mz_max, mz_min=mz_min
            )
            weights = spectrum.peaks.intensities ** peak_scaling

            # Find binned peaks present in peaks_vocab
            idx_in_vocab = [i for i, x in enumerate(doc) if x in peaks_vocab.keys()]
            idx_not_in_vocab = list(set(np.arange(len(doc))) - set(idx_in_vocab))

            doc_bow = [peaks_vocab[doc[i]] for i in idx_in_vocab]

            # TODO add missing weighted part!?!?
            peak_lists.append(list(zip(doc_bow, weights[idx_in_vocab])))
            if len(idx_in_vocab) == 0:
                missing_fractions.append(1.0)
            else:
                miss = np.nan_to_num(
                    np.sum(weights[idx_not_in_vocab]) / np.sum(weights)
                )
                missing_fractions.append(miss)

        return peak_lists, missing_fractions
