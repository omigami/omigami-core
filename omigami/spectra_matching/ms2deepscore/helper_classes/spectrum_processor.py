from typing import List, Dict, Optional, Union

import numpy as np
from matchms import Spectrum
from matchms.filtering import (
    select_by_mz,
    normalize_intensities,
)
from matchms.importing.load_from_json import as_spectrum
from matchmsextras.pubchem_lookup import pubchem_metadata_lookup

from omigami.common.progress_logger import TaskProgressLogger

INCORRECT_LAST_WORDS = [
    "M",
    "M?",
    "?",
    "M+2H/2",
    "MS34+Na",
    "M]",
    "Cat+M]",
    "Unk",
    "--",
]


class SpectrumProcessor:
    def process_spectra(
        self,
        spectra: Union[List[Dict], List[Spectrum]],
        process_reference_spectra: bool = True,
        progress_logger: TaskProgressLogger = None,
    ) -> List[Spectrum]:
        processed_spectra = []
        for i, spectrum in enumerate(spectra):
            if type(spectrum) == dict:
                spectrum = as_spectrum(spectrum)
            if spectrum is not None:
                spectrum = normalize_intensities(spectrum)
                spectrum = self._apply_ms2deepscore_filters(spectrum)
                if process_reference_spectra:
                    spectrum = self._check_inchikey(spectrum)

                if spectrum is not None:
                    processed_spectra.append(spectrum)

                if progress_logger:
                    progress_logger.log(i)

        return processed_spectra

    def _apply_ms2deepscore_filters(self, spectrum: Spectrum) -> Spectrum:
        """Remove spectra with less than 5 peaks with m/z values
        in the range between 10.0 and 1000.0 Da
        """
        spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
        spectrum = self._require_minimum_number_of_peaks(spectrum, n_required=5)
        return spectrum

    #  not currently used (see issue MLOPS-361)
    def _get_missing_inchis(
        self,
        spectrum: Spectrum,
    ) -> Optional[Spectrum]:
        if not spectrum:
            return None

        spectrum = self._process_compound_name(spectrum)
        return pubchem_metadata_lookup(spectrum)

    def _process_compound_name(self, spectrum: Spectrum) -> Spectrum:
        original_name = spectrum.get("compound_name")
        name = original_name.replace("F dial M", "")

        # Remove last word if likely not correct:
        if name.split(" ")[-1] in INCORRECT_LAST_WORDS:
            name = " ".join(name.split(" ")[:-1]).strip()

        if name != original_name:
            spectrum.set("compound_name", name)
        return spectrum

    @staticmethod
    def _check_inchikey(spectrum: Spectrum) -> Optional[Spectrum]:
        if spectrum:
            inchikey = spectrum.metadata.get("inchikey")
            if inchikey is not None and len(inchikey) > 13:
                if spectrum.get("inchi"):
                    cleaned_inchi = (
                        spectrum.get("inchi").replace('"', "").replace(" ", "")
                    )
                    spectrum.set("inchi", cleaned_inchi)
                    return spectrum

    def _select_ion_mode(self, spectrum: Spectrum) -> Optional[Spectrum]:
        # while we only support the already trained model, we just use the positive
        # spectra
        if spectrum:
            ion_mode = spectrum.metadata.get("ionmode")
            if ion_mode:
                if ion_mode.lower() == "positive":
                    return spectrum

    @staticmethod
    def _require_minimum_number_of_peaks(
        spectrum_in: Spectrum, n_required: int
    ) -> Optional[Spectrum]:
        if spectrum_in is None:
            return None

        spectrum = spectrum_in.clone()

        n_peaks = np.sum(spectrum.peaks.intensities > 0)
        if n_peaks < n_required:
            return None

        return spectrum
