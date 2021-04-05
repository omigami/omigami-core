import datetime
from typing import List

from prefect import task

from spec2vec_mlops.helper_classes.data_loader import DataLoader
from spec2vec_mlops.helper_classes.data_cleaner import DataCleaner
from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumStorer,
    SpectrumIDStorer,
)


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def clean_data_task(uri: str) -> List[str]:

    # load
    dl = DataLoader()
    data_loaded = dl.load_gnps_json(uri)

    # clean
    data_cleaner = DataCleaner()
    data_cleaned = [data_cleaner.clean_data(dl) for dl in data_loaded]

    # save
    storer = SpectrumStorer("spectrum_info")
    all_spectrum_ids = storer.store(data_cleaned)
    ids_storer = SpectrumIDStorer("spectrum_ids_info")
    ids_storer.store(all_spectrum_ids)
    return all_spectrum_ids
