from logging import getLogger
from typing import Union, List, Dict, Any, Tuple

from gensim.models import Word2Vec
from mlflow.pyfunc import PythonModel


log = getLogger(__name__)
SpectrumMatches = Dict[str, Dict[str, Any]]


class Predictor(PythonModel):
    def __init__(
        self,
        model: Word2Vec,
        n_decimals: int,
        intensity_weighting_power: Union[float, int],
        allowed_missing_percentage: Union[float, int],
        run_id: str = None,
    ):
        self.model = model
        self.n_decimals = n_decimals
        self.intensity_weighting_power = intensity_weighting_power
        self.allowed_missing_percentage = allowed_missing_percentage
        self.run_id = run_id

    def predict(
        self,
        context,
        data_input_and_parameters: Dict[str, Union[Dict, List]],
        mz_range: int = 1,
    ) -> float:
        """Match spectra from a json payload input with spectra having the highest similarity scores
        in the GNPS spectra library.
        Return a list matches of IDs and scores for each input spectrum.
        """
        log.info("Creating a prediction.")
        data_input, parameters = self._parse_input(data_input_and_parameters)
        log.info("Pre-processing data.")
        pre_processed_data = self._pre_process_data(data_input)

        log.info("Finishing prediction.")
        return 0

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    @staticmethod
    def _parse_input(
        data_input_and_parameters: Dict[str, Union[Dict, List]]
    ) -> Tuple[Union[dict, list, None], Union[dict, list, None, dict]]:
        if not isinstance(data_input_and_parameters, dict):
            data_input_and_parameters = data_input_and_parameters.tolist()

        parameters = (
            data_input_and_parameters.get("parameters")
            if data_input_and_parameters.get("parameters")
            else {}
        )
        data_input = data_input_and_parameters.get("data")
        return data_input, parameters

    @staticmethod
    def _pre_process_data(self, data_input: Dict[str, str]) -> List:
        embeddings = data_input
        return embeddings

