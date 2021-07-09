import multiprocessing
from joblib import Parallel, delayed
from typing import List, Dict, Optional, Union

from matchms import Spectrum
from matchms.filtering import (
    select_by_mz,
    require_minimum_number_of_peaks,
)
from matchms.importing.load_from_json import as_spectrum
from tqdm import tqdm

from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger
from omigami.spectrum_cleaner import SpectrumCleaner

NUM_CORES = multiprocessing.cpu_count() + 8


class SpectrumProcessor(SpectrumCleaner):
    def process_spectra(
        self,
        spectra: Union[List[Dict], List[Spectrum]],
        reference_spectra: bool = True,
        progress_logger: TaskProgressLogger = None,
    ) -> List[Spectrum]:
        processed_spectrum_dicts = Parallel(n_jobs=NUM_CORES, verbose=10000)(
            delayed(self._process_spectra)(spectrum, reference_spectra)
            for spectrum in tqdm(spectra)
        )
        processed_spectrum_dicts = [
            spectra for spectra in processed_spectrum_dicts if spectra is not None
        ]
        return processed_spectrum_dicts

    def _process_spectra(
        self,
        spectrum: Union[Dict, Spectrum],
        reference_spectra: bool = True,
    ):
        if type(spectrum) == dict:
            spectrum = as_spectrum(spectrum)
        if spectrum is not None:
            spectrum = self._apply_filters(spectrum)
            spectrum = self._apply_ms2deepscore_filters(spectrum)
            if reference_spectra:
                spectrum = self._select_ion_mode(spectrum)
                spectrum = self._harmonize_spectrum(spectrum)
                spectrum = self._convert_metadata(spectrum)
                spectrum = self._check_inchikey(spectrum)

                return spectrum

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
        if spectrum:
            inchikey = spectrum.metadata.get("inchikey")
            if inchikey is not None and len(inchikey) > 13:
                if spectrum.get("smiles") or spectrum.get("inchi"):
                    return spectrum

    def _select_ion_mode(self, spectrum: Spectrum) -> Optional[Spectrum]:
        # while we only support the already trained model, we just use the positive
        # spectra
        if spectrum:
            ion_mode = spectrum.metadata.get("ionmode")
            if ion_mode:
                if ion_mode.lower() == "positive":
                    return spectrum
