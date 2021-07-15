import logging
import numpy as np
from typing import List, Dict, Optional, Union

from matchms import Spectrum
from matchms.filtering import (
    select_by_mz,
    require_minimum_number_of_peaks,
)
from matchms.importing.load_from_json import as_spectrum
from matchms.utils import is_valid_inchikey

from omigami.spec2vec.helper_classes.progress_logger import TaskProgressLogger
from omigami.spectrum_cleaner import SpectrumCleaner

from matchmsextras.pubchem_lookup import (
    pubchem_name_search,
    likely_has_inchi,
    find_pubchem_inchi_match,
    pubchem_formula_search,
)


class SpectrumProcessor(SpectrumCleaner):
    def process_spectra(
        self,
        spectra: Union[List[Dict], List[Spectrum]],
        process_reference_spectra: bool = True,
        progress_logger: TaskProgressLogger = None,
    ) -> List[Spectrum]:
        processed_spectrum_dicts = []
        for i, spectrum in enumerate(spectra):
            if type(spectrum) == dict:
                spectrum = as_spectrum(spectrum)
            if spectrum is not None:
                spectrum = self._apply_filters(spectrum)
                spectrum = self._apply_ms2deepscore_filters(spectrum)
                if process_reference_spectra:
                    spectrum = self._select_ion_mode(spectrum)
                    spectrum = self._harmonize_spectrum(spectrum)
                    spectrum = self._convert_metadata(spectrum)
                    spectrum = self._run_missing_smiles_inchi_against_pubchem(spectrum)
                    spectrum = self._check_inchikey(spectrum)

                if spectrum is not None:
                    processed_spectrum_dicts.append(spectrum)

                if progress_logger:
                    progress_logger.log(i)

        return processed_spectrum_dicts

    @staticmethod
    def _apply_ms2deepscore_filters(spectrum: Spectrum) -> Spectrum:
        """Remove spectra with less than 5 peaks with m/z values
        in the range between 10.0 and 1000.0 Da
        """
        spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
        spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
        return spectrum

    def _run_missing_smiles_inchi_against_pubchem(
        self,
        spectrum: Spectrum,
    ) -> Optional[Spectrum]:
        if not spectrum:
            return None

        name_original = spectrum.get("compound_name")
        name = name_original.replace("F dial M", "")
        # Remove last word if likely not correct:
        if name.split(" ")[-1] in [
            "M",
            "M?",
            "?",
            "M+2H/2",
            "MS34+Na",
            "M]",
            "Cat+M]",
            "Unk",
            "--",
        ]:
            name = " ".join(name.split(" ")[:-1]).strip()
        if name != name_original:
            spectrum.set("compound_name", name)

        return self._pubchem_metadata_lookup(spectrum)

    @staticmethod
    def _check_inchikey(spectrum: Spectrum) -> Optional[Spectrum]:
        if spectrum:
            inchikey = spectrum.metadata.get("inchikey")
            if inchikey is not None and len(inchikey) > 13:
                if spectrum.get("inchi"):
                    cleaned_inchi = spectrum.get("inchi").replace('"', "")
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

    # TODO: after the PR is approved in matchmsextras we can delete the following 2
    #  methods and just call pubchem_metadata_lookup()
    def _pubchem_metadata_lookup(
        self,
        spectrum: Spectrum,
        name_search_depth=10,
        match_precursor_mz=False,
        formula_search=False,
        min_formula_length=6,
        formula_search_depth=25,
        verbose=1,
    ):
        spectrum = spectrum.clone()
        if is_valid_inchikey(spectrum.get("inchikey")):
            return spectrum

        def _plausible_name(compound_name):
            return isinstance(compound_name, str) and len(compound_name) > 4

        compound_name = spectrum.get("compound_name")
        if not _plausible_name(compound_name):
            return spectrum

        # Start pubchem search
        inchi = spectrum.get("inchi")
        parent_mass = spectrum.get("parent_mass")
        if isinstance(parent_mass, np.ndarray):
            parent_mass = parent_mass[0]
        formula = spectrum.get("formula")

        # 1) Search for matching compound name
        results_pubchem = pubchem_name_search(
            compound_name, name_search_depth=name_search_depth, verbose=verbose
        )

        if len(results_pubchem) > 0:

            # 1a) Search for matching inchi
            if likely_has_inchi(inchi):
                (
                    inchi_pubchem,
                    inchikey_pubchem,
                    smiles_pubchem,
                ) = find_pubchem_inchi_match(results_pubchem, inchi, verbose=verbose)
            # 1b) Search for matching parent mass
            if not likely_has_inchi(inchi) or inchikey_pubchem is None:
                (
                    inchi_pubchem,
                    inchikey_pubchem,
                    smiles_pubchem,
                ) = self._find_pubchem_mass_match(
                    results_pubchem,
                    parent_mass,
                    given_mass="parent mass",
                    verbose=verbose,
                )

            # 1c) Search for matching precursor mass (optional)
            if match_precursor_mz and inchikey_pubchem is None:
                precursor_mz = spectrum.get("precursor_mz")
                (
                    inchi_pubchem,
                    inchikey_pubchem,
                    smiles_pubchem,
                ) = self._find_pubchem_mass_match(
                    results_pubchem,
                    precursor_mz,
                    given_mass="precursor mass",
                    verbose=verbose,
                )

            if inchikey_pubchem is not None and inchi_pubchem is not None:
                logging.info("Matching compound name: %s", compound_name)
                if verbose >= 1:
                    print(f"Matching compound name: {compound_name}")
                spectrum.set("inchikey", inchikey_pubchem)
                spectrum.set("inchi", inchi_pubchem)
                spectrum.set("smiles", smiles_pubchem)
                return spectrum

            if verbose >= 2:
                print(f"No matches found for compound name: {compound_name}")

        # 2) Search for matching formula
        if formula_search and formula and len(formula) >= min_formula_length:
            results_pubchem = pubchem_formula_search(
                formula, formula_search_depth=formula_search_depth, verbose=verbose
            )

            if len(results_pubchem) > 0:

                # 2a) Search for matching inchi
                if likely_has_inchi(inchi):
                    (
                        inchi_pubchem,
                        inchikey_pubchem,
                        smiles_pubchem,
                    ) = find_pubchem_inchi_match(
                        results_pubchem, inchi, verbose=verbose
                    )
                # 2b) Search for matching parent mass
                if inchikey_pubchem is None:
                    (
                        inchi_pubchem,
                        inchikey_pubchem,
                        smiles_pubchem,
                    ) = self._find_pubchem_mass_match(
                        results_pubchem,
                        parent_mass,
                        given_mass="parent mass",
                        verbose=verbose,
                    )
                # 2c) Search for matching precursor mass (optional)
                if match_precursor_mz and inchikey_pubchem is None:
                    precursor_mz = spectrum.get("precursor_mz")
                    (
                        inchi_pubchem,
                        inchikey_pubchem,
                        smiles_pubchem,
                    ) = self._find_pubchem_mass_match(
                        results_pubchem,
                        precursor_mz,
                        given_mass="precursor mass",
                        verbose=verbose,
                    )
                if inchikey_pubchem is not None and inchi_pubchem is not None:
                    logging.info("Matching formula: %s", formula)
                    if verbose >= 1:
                        print(f"Matching formula: {formula}")
                    spectrum.set("inchikey", inchikey_pubchem)
                    spectrum.set("inchi", inchi_pubchem)
                    spectrum.set("smiles", smiles_pubchem)
                    return spectrum

                if verbose >= 2:
                    print(f"No matches found for formula: {formula}")

        return spectrum

    def _find_pubchem_mass_match(
        self,
        results_pubchem,
        parent_mass,
        mass_tolerance=2.0,
        given_mass="parent mass",
        verbose=1,
    ):

        inchi_pubchem = None
        inchikey_pubchem = None
        smiles_pubchem = None

        for result in results_pubchem:
            inchi_pubchem = '"' + result.inchi + '"'
            inchikey_pubchem = result.inchikey
            smiles_pubchem = result.isomeric_smiles
            if smiles_pubchem is None:
                smiles_pubchem = result.canonical_smiles

            pubchem_mass = float(results_pubchem[0].exact_mass)
            match_mass = np.abs(pubchem_mass - parent_mass) <= mass_tolerance

            if match_mass:
                logging.info(
                    "Matching molecular weight %s vs parent mass of %s",
                    str(np.round(pubchem_mass, 1)),
                    str(np.round(parent_mass, 1)),
                )
                if verbose >= 1:
                    print(
                        f"Matching molecular weight ({pubchem_mass:.1f} vs {given_mass} of {parent_mass:.1f})"
                    )
                break

        if not match_mass:
            inchi_pubchem = None
            inchikey_pubchem = None
            smiles_pubchem = None

            if verbose >= 2:
                print(f"No matches found for mass {parent_mass} Da")

        return inchi_pubchem, inchikey_pubchem, smiles_pubchem
