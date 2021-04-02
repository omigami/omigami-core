import datetime
from typing import List

from matchms import Spectrum
from prefect import task

from spec2vec_mlops.helper_classes.data_storer import SpectrumStorer, SpectrumIDStorer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def store_cleaned_data_task(data: List[Spectrum], out_dir: str, feast_core_url: str):
    storer = SpectrumStorer(out_dir, feast_core_url, feature_table_name="spectrum_info")
    all_spectrum_ids = storer.store_cleaned_data(data)
    ids_storer = SpectrumIDStorer(
        out_dir, feast_core_url, feature_table_name="spectrum_ids_info"
    )
    ids_storer.store_spectrum_ids(all_spectrum_ids)
