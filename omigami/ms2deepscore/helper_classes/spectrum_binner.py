from logging import Logger
from multiprocessing import Pool, cpu_count
from typing import List

import numpy as np
from matchms import Spectrum
from ms2deepscore import BinnedSpectrum, SpectrumBinner


class MS2DeepScoreSpectrumBinner:
    def __init__(self):
        self.spectrum_binner = SpectrumBinner(number_of_bins=10000)

    def bin_spectra(
        self,
        spectra: List[Spectrum],
        logger: Logger = None,
    ) -> List[BinnedSpectrum]:
        if not spectra:
            return []

        divisions = cpu_count()
        if logger:
            logger.info(f"Performing parallelized binning on {divisions} CPUs.")

        division_len = int(np.ceil(len(spectra) / divisions))

        spectra_divided = []
        for i in range(0, len(spectra), division_len):
            spectra_divided.append(spectra[i : min(i + division_len, len(spectra))])

        with Pool(divisions) as pool:
            binned_spectra = pool.map(
                self.spectrum_binner.fit_transform,
                spectra_divided,
            )

        binned_spectra = [item for sublist in binned_spectra for item in sublist]

        for binned_spectrum, spectrum in zip(binned_spectra, spectra):
            binned_spectrum.set("spectrum_id", spectrum.get("spectrum_id"))
            binned_spectrum.set("inchi", spectrum.get("inchi"))

        return binned_spectra
