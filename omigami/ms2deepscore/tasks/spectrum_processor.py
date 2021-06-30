from typing import List, Dict, Optional, Union

from matchms import Spectrum
from matchms.filtering import (
    select_by_mz,
    require_minimum_number_of_peaks,
)
from matchms.importing.load_from_json import as_spectrum

from omigami.spectrum_cleaner import SpectrumCleaner


class SpectrumProcessor(SpectrumCleaner):
    def process_spectra(self, spectra: Union[List[Dict], List[Spectrum]]):
        processed_spectrum_dicts = []
        for spectrum in spectra:
            if type(spectrum) == dict:
                spectrum = as_spectrum(spectrum)
            if spectrum is not None:
                spectrum = self._apply_filters(spectrum)
                spectrum = self._harmonize_spectrum(spectrum)
                spectrum = self._apply_ms2deepscore_filters(spectrum)

                if spectrum is not None:
                    processed_spectrum_dicts.append(spectrum)

        return processed_spectrum_dicts

    @staticmethod
    def _apply_ms2deepscore_filters(spectrum: Spectrum) -> Spectrum:
        """Remove spectra with less than 5 peaks with m/z values
        in the range between 10.0 and 1000.0 Da
        """
        spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
        spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
        return spectrum

    @staticmethod
    def _check_inchikey(spectrum: Spectrum) -> Optional[Spectrum]:
        inchikey = spectrum.get("inchikey")
        if inchikey is not None and len(inchikey) > 13:
            if spectrum.get("smiles") or spectrum.get("inchi"):
                return spectrum
