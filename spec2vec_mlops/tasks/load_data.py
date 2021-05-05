import datetime
from typing import Dict, List

from prefect import Task

from spec2vec_mlops.tasks.data_gateway import InputDataGateway


class LoadData(Task):
    def __init__(self, input_dgw: InputDataGateway, **kwargs):
        self._input_dgw = input_dgw
        super().__init__(
            max_retries=3, retry_delay=datetime.timedelta(seconds=10), **kwargs
        )

    def run(
        self, file_path: str = None, chunk_size: int = None
    ) -> List[List[Dict[str, str]]]:
        results = self._input_dgw.load_gnps(file_path)
        results_chunks = [
            results[i : i + chunk_size] for i in range(0, len(results), chunk_size)
        ]
        return results_chunks
