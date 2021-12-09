from logging import getLogger
from typing import List, Dict, Any

from mlflow.pyfunc import PythonModel

log = getLogger(__name__)
SpectrumMatches = Dict[str, Dict[str, Any]]


class Predictor(PythonModel):
    _run_id: str
    model: Any

    def __init__(self, dgw=None):
        self.dgw = dgw

    def predict(self, context, model_input):
        """Match spectra from a json payload input with spectra having the highest
        scores in the GNPS spectra library. Return a list matches of IDs and scores
        for each input spectrum.
        """
        raise NotImplementedError

    def _get_ref_ids_from_data_input(
        self, data_input: List[Dict[str, str]], mz_range: int = 1
    ) -> List[List[str]]:
        ref_spectrum_ids = []
        for i, spectrum in enumerate(data_input):
            precursor_mz = spectrum["Precursor_MZ"]
            min_mz, max_mz = (
                float(precursor_mz) - mz_range,
                float(precursor_mz) + mz_range,
            )
            ref_ids = self.dgw.get_spectrum_ids_within_range(min_mz, max_mz)
            ref_spectrum_ids.append(ref_ids)

        self._check_spectrum_refs(ref_spectrum_ids)
        return ref_spectrum_ids

    @staticmethod
    def _check_spectrum_refs(reference_spectra_ids: List[List[str]]):
        if [] in reference_spectra_ids:
            idx_null = [
                idx
                for idx, element in enumerate(reference_spectra_ids)
                if element == []
            ]
            raise RuntimeError(
                f"No data found from filtering with precursor MZ for spectra at indices {idx_null}. "
                f"Try increasing the mz_range filtering."
            )

    def _add_metadata(
        self, best_matches: Dict[str, SpectrumMatches]
    ) -> Dict[str, SpectrumMatches]:
        spectrum_ids = [key for match in best_matches.values() for key in match.keys()]

        spectra = self.dgw.read_spectra(set(spectrum_ids))
        spectra = {spectrum.metadata["spectrum_id"]: spectrum for spectrum in spectra}

        for matches in best_matches.values():
            for spectrum_id in matches.keys():
                matches[spectrum_id]["metadata"] = spectra[spectrum_id].metadata

        return best_matches

    def set_run_id(self, run_id: str):
        self._run_id = run_id
