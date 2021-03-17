from typing import List

from matchms import Spectrum

from spec2vec_mlops.custom_functions.pubchem_lookup import pubchem_metadata_lookup


class DataInputer:
    def __init__(self):
        pass

    def input_data(self, spectra_data: List[Spectrum]):
        """Complete metadata using pubchem lookup routines
        Try to complete and extend metadata based on looking for matches in PubChem.
        Found matches on PubChem will be accepted if:
         - Single matching extry for InchiKey (first 14 characters)
         - Single matching extry for Inchi
         - Single matching extry for Smiles
         - Matching name AND inchikey
         - Matching name AND inchi
         - Matching molecular formula AND inchi or inchikey
         - Single matching entry with same name AND molecular weight == parent mass.
         NOTE: This can be very slow
        """
        reference_spectrum_lookup = [
            pubchem_metadata_lookup(spectrum, name_search_depth=10)
            for spectrum in spectra_data
        ]


