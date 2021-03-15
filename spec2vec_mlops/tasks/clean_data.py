import datetime
from typing import List, Dict

from matchms import Spectrum
from matchms.importing.load_from_json import as_spectrum

from matchms.filtering import (
    default_filters,
    add_parent_mass,
    harmonize_undefined_inchikey,
    harmonize_undefined_inchi,
    harmonize_undefined_smiles,
    repair_inchi_inchikey_smiles,
    derive_inchi_from_smiles,
    derive_smiles_from_inchi,
    derive_inchikey_from_inchi,
)
from prefect import task


class DataCleaner:
    def __init__(self):
        pass

    def clean_data(self, data: List[Dict]) -> List[Spectrum]:
        # TODO: Paralelize with Dask
        spectra = [parsed for spec in data if (parsed := self._parse_data(spec))
                   is not None]
        return spectra

    def _parse_data(self, spectrum_dict: dict) -> Spectrum:
        spectrum = as_spectrum(spectrum_dict)
        if spectrum is not None:
            spectrum = self._apply_filters(spectrum)
            spectrum = self._harmonize_spectrum(spectrum)
            spectrum = self._convert_metadata(spectrum)

            return spectrum

    def _apply_filters(self, spectrum: Spectrum) -> Spectrum:
        spectrum = default_filters(spectrum)
        spectrum = add_parent_mass(spectrum)
        return spectrum

    def _harmonize_spectrum(self, spectrum: Spectrum) -> Spectrum:
        """
        Here, undefined entries will be harmonized (instead of having a huge variation
        of None,"", "N/A" etc.)
        The ``repair_inchi_inchikey_smiles`` function will correct misplaced metadata
        (e.g. inchikeys entered as inchi etc.) and harmonize the entry strings.
        """
        spectrum = harmonize_undefined_inchikey(spectrum)
        spectrum = harmonize_undefined_inchi(spectrum)
        spectrum = harmonize_undefined_smiles(spectrum)
        spectrum = repair_inchi_inchikey_smiles(spectrum)
        return spectrum

    def _convert_metadata(self, spectrum: Spectrum) -> Spectrum:
        """
        Where possible (and necessary, i.e. missing): Convert between smiles, inchi,
        inchikey to complete metadata. This is done using functions from rdkit.
        """
        spectrum = derive_inchi_from_smiles(spectrum)
        spectrum = derive_smiles_from_inchi(spectrum)
        spectrum = derive_inchikey_from_inchi(spectrum)
        return spectrum


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def clean_data(spectra_data: List[Dict]):
    data_cleaner = DataCleaner()
    results = data_cleaner.clean_data(spectra_data)
    return results
