from dataclasses import dataclass
from typing import Set

from prefect import Task

from omigami.config import IonModes
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.helper_classes.tanimoto_score_calculator import (
    TanimotoScoreCalculator,
)
from omigami.utils import merge_prefect_task_configs


@dataclass
class CalculateTanimotoScoreParameters:
    scores_output_path: str
    ion_mode: IonModes
    n_bits: int = 2048
    decimals: int = 5


class CalculateTanimotoScore(Task):
    def __init__(
        self,
        spectrum_gtw: MS2DeepScoreRedisSpectrumDataGateway,
        parameters: CalculateTanimotoScoreParameters,
        **kwargs,
    ):
        self._spectrum_gtw = spectrum_gtw
        self._scores_output_path = parameters.scores_output_path
        self._ion_mode = parameters.ion_mode
        self._n_bits = parameters.n_bits
        self._decimals = parameters.decimals
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, spectrum_ids: Set[str] = None) -> str:
        """
        Prefect task to calculate the Tanimoto Scores for all pairs of binned spectra.
        Calculated scores are saved to S3 filesystem.

        Parameters
        ----------
        spectrum_ids: Set[str]
            Set of spectrum_ids to calculate Tanitamo Scores

        Returns
        -------
        path: str
            path where scores are saved

        """
        self.logger.info(f"Calculating the Tanimoto Scores")
        calculator = TanimotoScoreCalculator(
            spectrum_dgw=self._spectrum_gtw,
            ion_mode=self._ion_mode,
            n_bits=self._n_bits,
        )
        path = calculator.calculate(
            list(spectrum_ids),
            self._scores_output_path,
            self.logger,
        )
        return path
