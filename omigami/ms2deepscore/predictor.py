from logging import getLogger
from typing import Union, List, Dict, Tuple

import numpy as np
from matchms.Spectrum import Spectrum
from mlflow.pyfunc import PythonModel
from ms2deepscore.models import load_model as ms2deepscore_load_model
from ms2deepscore import MS2DeepScore


log = getLogger(__name__)


class Predictor(PythonModel):
    def __init__(
        self,
        run_id: str = None,
    ):
        self.run_id = run_id

    def load_context(self, context):
        model_path = context.artifacts["ms2deepscore_model_path"]
        try:
            log.info(f"Loading model from {model_path}")
            siamese_model = ms2deepscore_load_model(model_path)
            self.model = MS2DeepScore(siamese_model)
        except FileNotFoundError:
            log.error(f"Could not find MS2DeepScore model in {model_path}")

    def predict(
        self,
        context,
        data_input: Dict[str, List],
    ) -> Dict:
        """Calculate the MS2DeepScore similarity between a reference and a query spectrum,
        which are extracted from the dict provided in data_input. An example of data_input
        is as follows:

        {
            "data": [
                {
                    "intensities": reference_intensities,
                    "mz": reference_mz,
                },
                {
                    "intensities": query_intensities,
                    "mz": query_mz,
                },
            ],
            "parameters": {"n_best": int, "include_metadata": List[str]}
        }

        Returns
        -------
        ms2ds_similarity
            MS2DeepScore similarity score.
        """
        log.info("Creating a prediction.")
        data_input, parameters = self._parse_input(data_input)
        spectra = self._clean_spectra(data_input)

        reference = Spectrum(**spectra[0])
        query = Spectrum(**spectra[1])

        score = self.model.pair(reference, query)

        log.info("Finishing prediction.")
        return {"score": score}

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    @staticmethod
    def _parse_input(
        data_input_and_parameters: Dict[str, Union[Dict, List]]
    ) -> Tuple[Union[dict, list, None], Union[dict, list, None, dict]]:
        if not isinstance(data_input_and_parameters, dict):
            data_input_and_parameters = data_input_and_parameters.tolist()

        data_input = data_input_and_parameters.get("data")
        parameters = data_input_and_parameters.get("parameters")

        return data_input, parameters

    def _clean_spectra(self, data_input: List) -> List:
        return [
            {
                "intensities": np.asarray(spectrum["intensities"]),
                "mz": np.asarray(spectrum["mz"]),
            }
            for spectrum in data_input
        ]
