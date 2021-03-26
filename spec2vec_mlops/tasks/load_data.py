import datetime
from typing import Dict, List

from prefect import task

from spec2vec_mlops.helper_classes.data_loader import DataLoader


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def load_data_task(uri) -> List[Dict[str, str]]:
    dl = DataLoader()
    results = dl.load_gnps_json(uri)
    return results
