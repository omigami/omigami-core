from logging import getLogger
from typing import Union, List, Dict

from matchms.Spectrum import Spectrum
from mlflow.pyfunc import PythonModel
from ms2deepscore import MS2DeepScore

log = getLogger(__name__)


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
        data_input: Dict[str, List],
    ) -> float:
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
        }

        Returns
        -------
        ms2ds_similarity
            MS2DeepScore similarity score.
        """
        log.info("Creating a prediction.")
        data_input = self._parse_input(data_input)

        reference = Spectrum(**data_input[0])
        query = Spectrum(**data_input[1])

        score = self.model.pair(reference, query)

        log.info("Finishing prediction.")
        return score

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    @staticmethod
    def _parse_input(data_input_and_parameters: Dict[str, Union[Dict, List]]) -> list:
        if not isinstance(data_input_and_parameters, dict):
            data_input_and_parameters = data_input_and_parameters.tolist()

        data_input = data_input_and_parameters.get("data")

        return data_input
