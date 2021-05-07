from typing import Dict, List

from prefect import Task, task

from spec2vec_mlops.tasks.config import DEFAULT_CONFIG
from spec2vec_mlops.tasks.data_gateway import InputDataGateway


class LoadData(Task):
    def __init__(self, input_dgw: InputDataGateway, **kwargs):
        self._input_dgw = input_dgw
        super().__init__(**DEFAULT_CONFIG, **kwargs)

    def run(
        self, file_path: str = None, chunk_size: int = None
    ) -> List[List[Dict[str, str]]]:
        results = self._input_dgw.load_gnps(file_path)
        results_chunks = [
            results[i : i + chunk_size] for i in range(0, len(results), chunk_size)
        ]
        return results_chunks


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def load_data_task(
    file_path: DRPath,
    ionmode: str,
    chunksize: int = 1000,
    n_decimals: int = 2,
    min_peaks: int = 5,
    skip_if_exists: bool = True,
) -> List[List[str]]:
    dl = DataLoader(file_path, n_decimals, min_peaks)
    results = dl.load_gnps_json(ionmode=ionmode, skip_if_exists=skip_if_exists)
    results_chunks = [
        results[i : i + chunksize] for i in range(0, len(results), chunksize)
    ]
    return results_chunks
