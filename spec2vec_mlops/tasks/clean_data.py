import datetime
from typing import List, Dict

import prefect
from prefect import task

from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.gateways.redis_gateway import RedisDataGateway
from spec2vec_mlops.helper_classes.data_cleaner import DataCleaner


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def clean_data_task(
    spectra_data_chunks: List[Dict], n_decimals: int, skip_if_exist: bool = True
) -> List[str]:
    logger = prefect.context.get("logger")
    beg = datetime.datetime.now()

    dgw = RedisDataGateway()

    if skip_if_exist:
        spectrum_ids = [sp.get("spectrum_id") for sp in spectra_data_chunks]
        spectrum_ids = dgw.list_spectra_not_exist(spectrum_ids)
        spectra_data_chunks = [
            sp for sp in spectra_data_chunks if sp.get("spectrum_id") in spectrum_ids
        ]

    data_cleaner = DataCleaner()
    cleaned_data = [
        data_cleaner.clean_data(spectra_data) for spectra_data in spectra_data_chunks
    ]
    cleaned_data = [spectrum for spectrum in cleaned_data if spectrum]
    spectra_data = [
        SpectrumDocumentData(spectrum, n_decimals) for spectrum in cleaned_data
    ]

    dgw.write_spectrum_documents(spectra_data)
    logger.info(
        f"Clean and convert {len(spectra_data)} spectrum documents in {datetime.datetime.now() - beg} hours."
    )
    return [sp.spectrum_id for sp in spectra_data]
