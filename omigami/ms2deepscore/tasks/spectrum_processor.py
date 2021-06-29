from typing import List, Dict

from matchms.filtering import (
    select_by_mz,
    require_minimum_number_of_peaks,
)
from matchms.importing.load_from_json import as_spectrum

from omigami.spectrum_cleaner import SpectrumCleaner


class SpectrumProcessor(SpectrumCleaner):
    def process_spectrum(self, spectrum_dicts: List[Dict], min_peaks=None):
        processed_spectrum_dicts = []
        for i, spectrum_dict in enumerate(spectrum_dicts):
            spectrum = as_spectrum(spectrum_dict)
            if spectrum is not None and len(spectrum.peaks.mz) > min_peaks:
                processed_spectrum = self._convert_metadata(
                    self._harmonize_spectrum(self._apply_filters(spectrum))
                )
                processed_spectrum_dicts.append(processed_spectrum)
        return processed_spectrum_dicts

    @staticmethod
    def _minimal_processing(spectrum):
        """We here remove spectra with less than 5 peaks with m/z values
        in the range between 10.0 and 1000.0 Da
        """
        spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
        spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
        return spectrum
