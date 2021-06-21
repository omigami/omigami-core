from typing import Dict, List

from matchms import Spectrum
from matchms.filtering import (
    default_filters,
    add_parent_mass,
    normalize_intensities,
    harmonize_undefined_inchikey,
    harmonize_undefined_inchi,
    harmonize_undefined_smiles,
    repair_inchi_inchikey_smiles,
    derive_inchi_from_smiles,
    derive_smiles_from_inchi,
    derive_inchikey_from_inchi,
)
from matchms.importing.load_from_json import as_spectrum

from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger


class SpectrumProcessor:
    def create_documents(
        self,
        spectrum_dicts: List[Dict],
        min_peaks: int = 0,
        n_decimals: int = 2,
        progress_logger: TaskProgressLogger = None,
    ) -> List[SpectrumDocumentData]:
        # TODO: there is something wrong with this code compared to what we had before.
        # need to investigate.
        documents = []
        for i, spectrum_dict in enumerate(spectrum_dicts):
            spectrum = as_spectrum(spectrum_dict)
            if spectrum is not None and len(spectrum.peaks.mz) > min_peaks:
                processed_spectrum = self._convert_metadata(
                    self._harmonize_spectrum(self._apply_filters(spectrum))
                )
                doc = SpectrumDocumentData(processed_spectrum, n_decimals)
                # cz: I'm not sure why this check is needed. looks like some bad implementation
                # somewhere. We should try to investigate this if we have the time
                if doc.spectrum and doc.document:
                    documents.append(doc)

                if progress_logger:
                    progress_logger.log(i)

        return documents

    @staticmethod
    def _apply_filters(spectrum: Spectrum) -> Spectrum:
        """Applies a collection of filters to normalize data, like convert str to int"""
        spectrum = default_filters(spectrum)
        spectrum = add_parent_mass(spectrum)
        spectrum = normalize_intensities(spectrum)
        return spectrum

    @staticmethod
    def _harmonize_spectrum(spectrum: Spectrum) -> Spectrum:
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

    @staticmethod
    def _convert_metadata(spectrum: Spectrum) -> Spectrum:
        """
        Where possible (and necessary, i.e. missing): Convert between smiles, inchi,
        inchikey to complete metadata. This is done using functions from rdkit.
        """
        spectrum = derive_inchi_from_smiles(spectrum)
        spectrum = derive_smiles_from_inchi(spectrum)
        spectrum = derive_inchikey_from_inchi(spectrum)
        return spectrum