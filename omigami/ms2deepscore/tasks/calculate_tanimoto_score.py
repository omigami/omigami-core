from typing import Dict

from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.helper_classes.tanimoto_score_calculator import (
    TanimotoScoreCalculator,
)
from prefect import Task
from omigami.utils import merge_prefect_task_configs


class CalculateTanimotoScore(Task):
    """
    Prefect task to calculate the Tanimoto Scores for all pairs of binned spectra
    """

    def __init__(
        self,
        n_bits: int = 2048,
        **kwargs,
    ):
        self._n_bits = n_bits
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self) -> Dict[str, Dict[str, float]]:
        self.logger.info(f"Calculating the Tanimoto Scores")
        calculator = TanimotoScoreCalculator(
            spectrum_dgw=MS2DeepScoreRedisSpectrumDataGateway()
        )
        scores = calculator.calculate(n_bits=self._n_bits)
        return scores.to_dict()
