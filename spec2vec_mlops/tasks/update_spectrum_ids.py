import datetime
from typing import List

from prefect import task

from spec2vec_mlops.helper_classes.storer_classes import SpectrumIDStorer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def update_spectrum_ids_task(spectrum_ids_chunks: List[List[str]]) -> List[List[str]]:
    spectrum_ids = [id for chunk in spectrum_ids_chunks for id in chunk]
    ids_storer = SpectrumIDStorer("spectrum_ids_info")
    ids_storer.store(spectrum_ids)
    ids_storer.store_online()
    return spectrum_ids_chunks
