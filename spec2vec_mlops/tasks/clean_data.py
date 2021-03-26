import datetime
from typing import Dict

from prefect import task

from spec2vec_mlops.helper_classes.data_cleaner import DataCleaner


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def clean_data_task(spectra_data: Dict):
    data_cleaner = DataCleaner()
    results = data_cleaner.clean_data(spectra_data)
    return results
