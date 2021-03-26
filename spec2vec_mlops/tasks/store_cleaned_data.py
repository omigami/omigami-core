import datetime
from typing import List
from matchms import Spectrum
from prefect import task
from spec2vec_mlops.helper_classes.data_storer import DataStorer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def store_cleaned_data_task(data: List[Spectrum], out_dir: str, feast_core_url: str):
    ds = DataStorer(out_dir, feast_core_url)
    ds.store_cleaned_data(data)
