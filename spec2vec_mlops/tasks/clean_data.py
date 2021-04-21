import datetime
from typing import List, Dict

from prefect import task

from spec2vec_mlops.helper_classes.data_cleaner import DataCleaner
from spec2vec_mlops.helper_classes.storer_classes import SpectrumStorer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def clean_data_task(spectra_data_chunks: List[Dict]) -> List[str]:
    data_cleaner = DataCleaner()
    cleaned_data = [
        data_cleaner.clean_data(spectra_data) for spectra_data in spectra_data_chunks
    ]
    cleaned_data = [spectrum for spectrum in cleaned_data if spectrum]
    storer = SpectrumStorer("spectrum_info")
    spectrum_ids = storer.store(cleaned_data)
    return spectrum_ids
