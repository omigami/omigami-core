import datetime
from pathlib import Path
from typing import Dict, List

from prefect import task

from spec2vec_mlops.helper_classes.data_loader import DataLoader


def load_data_task(file_path: Path, chunksize: int = 1000) -> List[Dict[str, str]]:
    dl = DataLoader(file_path)
    results = dl.load_gnps_json()
    results_chunks = [
        results[i : i + chunksize] for i in range(0, len(results), chunksize)
    ]
    return results_chunks
