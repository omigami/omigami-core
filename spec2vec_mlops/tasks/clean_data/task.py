import datetime
from typing import List

import prefect
from prefect import Task

from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.tasks.clean_data.clean_data import DataCleaner
from spec2vec_mlops.tasks.config import DEFAULT_CONFIG
from spec2vec_mlops.tasks.data_gateway import SpectrumDataGateway


class CleanData(Task):
    def __init__(
        self,
        spectrum_dgw: SpectrumDataGateway,
        n_decimals: int,
        skip_if_exists: bool = True,
    ):
        self._redis_dgw = spectrum_dgw
        self._n_decimals = n_decimals
        self._skip_if_exists = skip_if_exists
        super().__init__(**DEFAULT_CONFIG)

    def run(self, spectra_data_chunks: List[dict] = None) -> List[str]:
        logger = prefect.context.get("logger")
        start = datetime.datetime.now()

        if self._skip_if_exists:
            spectrum_ids = [sp.get("spectrum_id") for sp in spectra_data_chunks]
            spectrum_ids = self._redis_dgw.list_spectra_not_exist(spectrum_ids)
            spectra_data_chunks = [
                sp
                for sp in spectra_data_chunks
                if sp.get("spectrum_id") in spectrum_ids
            ]

        data_cleaner = DataCleaner()
        cleaned_data = [
            data_cleaner.clean_data(spectra_data)
            for spectra_data in spectra_data_chunks
        ]
        cleaned_data = [spectrum for spectrum in cleaned_data if spectrum]
        spectra_data = [
            SpectrumDocumentData(spectrum, self._n_decimals)
            for spectrum in cleaned_data
        ]

        self._redis_dgw.write_spectrum_documents(spectra_data)
        logger.info(
            f"Clean and convert {len(spectra_data)} spectrum documents in "
            f"{datetime.datetime.now() - start} hours."
        )
        return [sp.spectrum_id for sp in spectra_data]
