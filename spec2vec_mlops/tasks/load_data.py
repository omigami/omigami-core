import datetime
from typing import List

from drfs import DRPath
from prefect import task

from spec2vec_mlops.helper_classes.data_loader import DataLoader


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
