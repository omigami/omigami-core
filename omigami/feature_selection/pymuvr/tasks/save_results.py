from prefect import Task

from omigami.feature_selection.pymuvr.entities.feature_selection_results import (
    FeatureSelectionResult,
)
from omigami.utils import merge_prefect_task_configs


class SaveResults(Task):
    def __init__(self, **kwargs):
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, feature_selection_result: FeatureSelectionResult = None) -> None:
        """
        Task to save pymuvr feature selection results to the filesystem

        Parameters
        ----------
        feature_selection_result: FeatureSelectionResult

        Returns
        -------

        """
        pass
