from dataclasses import dataclass
from typing import List, Dict, Optional

from drfs import DRPath
from matchms import Spectrum
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
from matchms.importing.load_from_json import as_spectrum
from prefect import Task

from omigami.spectra_matching.storage import DataGateway
from omigami.utils import merge_prefect_task_configs


@dataclass
class CleanRawSpectraParameters:
    """
    Parameters to determine aspects of the CleanRawSpectra task

    output_directory:
        Directory where the cleaned spectra will be saved
    """

    output_directory: str


class CleanRawSpectra(Task):
    """
    Prefect task to save the raw spectra passed to it.
    """

    def __init__(
        self,
        fs_dgw: DataGateway,
        parameters: CleanRawSpectraParameters,
        **kwargs,
    ):
        self._fs_dgw = fs_dgw
        self._output_directory = parameters.output_directory
        self._spectrum_cleaner = SpectrumCleaner()
        config = merge_prefect_task_configs(kwargs)

        super().__init__(**config)

    def run(self, raw_spectra_path: str = None) -> str:
        """
        Loads a json file containing spectra from GNPS and applies a series of cleaning
        steps to the spectra. Some spectra might be filtered out in the process.
        Saves the spectra to the file system.

        Parameters:
        ----------
        raw_spectra_path: str
            A string leading to a json datafile containing spectrum data

        Returns:
        --------
        A path to the saved cleaned spectra

        """
        self.logger.info(f"Loading spectra from {raw_spectra_path}.")
        output_path = f"{self._output_directory}/{DRPath(raw_spectra_path).stem}.pickle"

        if DRPath(output_path).exists():
            self.logger.info(f"Using cached result at {output_path}")
            return output_path

        spectra = self._fs_dgw.load_spectrum(raw_spectra_path)
        spectrum_ids = [sp["spectrum_id"] for sp in spectra]

        self.logger.info(f"Cleaning {len(spectrum_ids)} spectra.")
        clean_spectra = self._spectrum_cleaner.clean(spectra)
        clean_spectrum_ids = [sp.metadata["spectrum_id"] for sp in clean_spectra]
        self.logger.info(f"There are {len(clean_spectrum_ids)} spectra after cleaning.")

        self.logger.info(f"Saving cleaned spectra to file {output_path}.")
        self._fs_dgw.serialize_to_file(output_path, clean_spectra)
        return output_path


class SpectrumCleaner:
    def clean(self, spectra: List[Dict]) -> List[Spectrum]:
        processed_spectra = []
        for spectrum in spectra:
            spectrum = as_spectrum(spectrum)
            spectrum = self._common_cleaning(spectrum)
            if spectrum is not None:
                if any(spectrum.peaks.intensities > 0):
                    processed_spectra.append(spectrum)
        return processed_spectra

    def _common_cleaning(self, spectrum: Spectrum) -> Optional[Spectrum]:
        spectrum = self._apply_filters(spectrum)

        if spectrum is None:
            return spectrum

        # On GNPS data, parent_mass is not present. Instead, if available, the Exact Mass
        # field should be used. In this context, they are synonyms
        spectrum.metadata["parent_mass"] = (
            spectrum.metadata["exactmass"]
            if float(spectrum.metadata.get("exactmass", 0)) > 0
            else None
        )

        spectrum = add_parent_mass(spectrum)
        spectrum = self._harmonize_spectrum(spectrum)
        spectrum = self._convert_metadata(spectrum)
        return spectrum

    def _apply_filters(self, spectrum: Spectrum) -> Spectrum:
        """Applies a collection of filters to normalize data, like convert str to int"""
        spectrum = default_filters(spectrum)
        spectrum = self._filter_negative_intensities(spectrum)
        return spectrum

    @staticmethod
    def _filter_negative_intensities(spectrum: Spectrum) -> Optional[Spectrum]:
        """Will return None if the given Spectrum's intensity has negative values."""

        if spectrum and any(spectrum.peaks.intensities < 0):
            return None

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
