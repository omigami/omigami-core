from dataclasses import dataclass
from typing import Set

from prefect import Task

from omigami.spectra_matching.ms2deepscore.helper_classes.tanimoto_score_calculator import (
    TanimotoScoreCalculator,
)

from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.utils import merge_prefect_task_configs


@dataclass
class CalculateTanimotoScoreParameters:
    scores_output_path: str
    binned_spectra_path: str
    n_bits: int = 2048
    decimals: int = 5


class CalculateTanimotoScore(Task):
    def __init__(
        self,
        fs_dgw: MS2DeepScoreFSDataGateway,
        parameters: CalculateTanimotoScoreParameters,
        **kwargs,
    ):
        self._fs_dgw = fs_dgw
        self._scores_output_path = parameters.scores_output_path
        self._binned_spectra_path = parameters.binned_spectra_path
        self._n_bits = parameters.n_bits
        self._decimals = parameters.decimals
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, spectrum_ids: Set[str]) -> str:
        """
        Prefect task to calculate the Tanimoto Scores for all pairs of binned spectra.
        Calculated scores are saved to filesystem.

        Parameters
        ----------
        spectrum_ids: Set[str]
            Set of spectrum_ids to calculate Tanitamo Scores. Only here to create depenency on previous flow task.

        Returns
        -------
        path: str
            path where scores are saved

        """
        self.logger.info(f"Calculating the Tanimoto Scores")
        calculator = TanimotoScoreCalculator(
            fs_dgw=self._fs_dgw,
            n_bits=self._n_bits,
            binned_spectra_path=self._binned_spectra_path,
        )
        path = calculator.calculate(
            self._scores_output_path,
            self.logger,
        )
        return path
