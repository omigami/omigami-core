import dask.dataframe as dd
from prefect import Task

from omigami.utils import merge_prefect_task_configs


class LoadDataset(Task):
    def __init__(self, **kwargs):
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, dataset_path: str = None) -> dd.DataFrame:
        """
        Task to load dataset from given path

        Parameters
        ----------
        dataset_path: str
            path to load dataset from

        Returns
        -------
        dd.DataFrame

        """
        pass
