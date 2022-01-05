from dataclasses import dataclass
from typing import Literal, Union

import pandas as pd
from prefect import Task
from py_muvr.data_structures import MetricFunction, InputEstimator

from omigami.feature_selection.pymuvr.entities.feature_selection_results import (
    FeatureSelectionResult,
)
from omigami.utils import merge_prefect_task_configs

_ESTIMATOR = Literal["RFC", "XGBC", "PLSC", "PLSR"]


@dataclass
class RunPymuvrParameters:
    n_outer: int
    n_inner: int
    estimator: Union[_ESTIMATOR, InputEstimator]
    metric: Union[str, MetricFunction]
    random_state: int
    n_repetitions: int = 8
    features_dropout_rate: float = 0.05
    robust_minimum: float = 0.05


class RunPymuvr(Task):
    def __init__(self, parameters: RunPymuvrParameters, **kwargs):
        config = merge_prefect_task_configs(kwargs)
        self._n_repetitions = parameters.n_repetitions
        self._n_outer = parameters.n_outer
        self._n_inner = parameters.n_inner
        self._estimator = parameters.estimator
        self._metric = parameters.metric
        self._features_dropout_rate = parameters.features_dropout_rate
        self._robust_minimum = parameters.robust_minimum
        self._random_state = parameters.random_state
        super().__init__(**config)

    def run(self, dataset: pd.DataFrame = None) -> FeatureSelectionResult:
        """
        Task to run pymuvr feature selection algorithm

        Parameters
        ----------
        dataset: TrainingDataset
            dataset to use for pymuvr training

        Returns
        -------
        FeatureSelectionResult
            object containing feature name and rank

        """
        pass
