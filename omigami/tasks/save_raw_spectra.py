from prefect import Task
from omigami.utils import create_prefect_result_from_path, merge_prefect_task_configs


class SaveRawSpectra(Task):
    def __init__(self, **kwargs):

        config = merge_prefect_task_configs(kwargs)
        super().__init__(
            **config,
            checkpoint=True,
        )

    def run(self):

        return False
