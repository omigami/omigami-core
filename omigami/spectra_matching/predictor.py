from logging import getLogger
from typing import List, Dict, Any

import flask
from flask import jsonify
from mlflow.pyfunc import PythonModel

from omigami.spectra_matching.storage import RedisSpectrumDataGateway

log = getLogger(__name__)
SpectrumMatches = Dict[str, Dict[str, Any]]


class SpectraMatchingError(Exception):
    status_code = 404

    def __init__(self, message, application_error_code, http_status_code):
        Exception.__init__(self, message)
        self.message = message
        if http_status_code is not None:
            self.status_code = http_status_code
        self.application_error_code = application_error_code

    def to_dict(self):
        res = {
            "status": {
                "status": self.status_code,
                "message": self.message,
                "app_code": self.application_error_code,
            }
        }
        return res

    @property
    def _repr(self):
        return f"SpectraMatchingError {self.status_code}: '{self.message}'"

    def __repr__(self):
        return self._repr

    def __str__(self):
        return self._repr


class Predictor(PythonModel):
    _run_id: str
    model: Any

    def __init__(self, dgw: RedisSpectrumDataGateway = None):
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
            if len(ref_ids) == 0:
                raise RuntimeError(
                    f"No data found from filtering with precursor MZ for precursor MZ {precursor_mz}. "
                    f"and mz_range {mz_range}. Try increasing the mz_range filtering."
                )

            ref_spectrum_ids.append(ref_ids)

        return ref_spectrum_ids

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

    model_error_handler = flask.Blueprint("error_handlers", __name__)


@Predictor.model_error_handler.app_errorhandler(SpectraMatchingError)
def handle_custom_error(error: SpectraMatchingError):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response
