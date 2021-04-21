import datetime
from typing import List

from prefect import task

from spec2vec_mlops.helper_classes.base_storer import BaseStorer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def update_feast_online_task(
    storer_classes: List[BaseStorer], spectrum_ids_chunks: List[List[str]]
) -> List[List[str]]:
    for storer_class in storer_classes:
        storer_class.store_online()
    return spectrum_ids_chunks
