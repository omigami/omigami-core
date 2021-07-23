from dataclasses import dataclass
from typing import List, Set

import prefect
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.helper_classes.tanimoto_score_calculator import (
    TanimotoScoreCalculator,
)
from omigami.utils import merge_prefect_task_configs
from prefect import Task


@dataclass
class CalculateTanimotoScoreParameters:
    scores_output_path: str
    n_bits: int = 2048
    decimals: int = 5


class CalculateTanimotoScore(Task):
    """
    Prefect task to calculate the Tanimoto Scores for all pairs of binned spectra
    """

    def __init__(
        self,
        parameters: CalculateTanimotoScoreParameters,
        **kwargs,
    ):
        self._scores_output_path = parameters.scores_output_path
        self._n_bits = parameters.n_bits
        self._decimals = parameters.decimals
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config, trigger=prefect.triggers.all_successful)

    def run(self, spectrum_ids_chunks: List[Set[str]] = None) -> str:
        flattened_ids = [item for elem in spectrum_ids_chunks for item in elem]
        self.logger.info(f"Calculating the Tanimoto Scores")
        calculator = TanimotoScoreCalculator(
            spectrum_dgw=MS2DeepScoreRedisSpectrumDataGateway(), n_bits=self._n_bits
        )
        path = calculator.calculate(
            flattened_ids,
            self._scores_output_path,
        )
        return path
