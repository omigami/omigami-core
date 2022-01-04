import pandas as pd
from prefect import Task

from omigami.feature_selection.pymuvr.entities.feature_selection_results import (
    FeatureSelectionResult,
)
from omigami.utils import merge_prefect_task_configs


class RunPymuvr(Task):
    def __init__(self, **kwargs):
        config = merge_prefect_task_configs(kwargs)
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
