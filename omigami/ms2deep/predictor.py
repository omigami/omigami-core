from logging import getLogger
from typing import Union, List, Dict, Any, Tuple

from matchms.Spectrum import Spectrum
from mlflow.pyfunc import PythonModel
from ms2deepscore import MS2DeepScore


log = getLogger(__name__)
SpectrumMatches = Dict[str, Dict[str, Any]]


class Predictor(PythonModel):
    def __init__(
        self,
        model: MS2DeepScore,
        run_id: str = None,
    ):
        self.model = model
        self.run_id = run_id

    def predict(
        self,
        context,
        data_input_and_parameters: Dict[str, Union[Dict, List]],
    ) -> float:
        log.info("Creating a prediction.")
        data_input, parameters = self._parse_input(data_input_and_parameters)

        spectrum_a = Spectrum(**data_input[0])
        spectrum_b = Spectrum(**data_input[1])

        score = self.score(spectrum_a, spectrum_b)

        log.info("Finishing prediction.")
        return score

    def score(self, spectrum_a: Spectrum, spectrum_b: Spectrum) -> float:
        score = self.model.pair(spectrum_a, spectrum_b)
        return score

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    @staticmethod
    def _parse_input(
        data_input_and_parameters: Dict[str, Union[Dict, List]]
    ) -> Tuple[Union[dict, list, None], Union[dict, list, None, dict]]:
        if not isinstance(data_input_and_parameters, dict):
            data_input_and_parameters = data_input_and_parameters.tolist()

        parameters = data_input_and_parameters.get("parameters", {})
        data_input = data_input_and_parameters.get("data")

        return data_input, parameters
