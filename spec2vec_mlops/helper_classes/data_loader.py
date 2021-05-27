import datetime
import logging
from concurrent import futures
from typing import Dict, List, Any

import ijson
import numpy as np
from drfs import DRPath
from drfs.filesystems import get_fs

from spec2vec_mlops.config import default_configs
from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway
from spec2vec_mlops.tasks.process_spectrum import SpectrumProcessor

KEYS = default_configs["gnps_json"]["necessary_keys"]


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self,
        file_path: DRPath,
        n_decimals: int = 2,
        min_peaks: int = 5,
        n_workers: int = 4,
    ):
        self.fs = get_fs(str(file_path))
        self.file_path = file_path
        self.n_decimals = n_decimals
        self.min_peaks = min_peaks
        self.dgw = RedisSpectrumDataGateway()
        self.n_workers = n_workers

    def load_gnps_json(
        self, ionmode: str = None, skip_if_exists: bool = True, chunksize: int = 10000
    ) -> List[str]:
        beg = datetime.datetime.now()
        with self.fs.open(self.file_path, "rb") as f:
            items = ijson.items(f, "item", multiple_values=True)
            n_loaded = 0
            chunk = []
            for item in items:
                if item.get("Ion_Mode").lower() == ionmode:
                    spectrum_id = item.get("SpectrumID")
                    if (
                        not skip_if_exists
                        or spectrum_id in self.dgw.list_spectra_not_exist([spectrum_id])
                    ):
                        chunk.append(item)
                        if len(chunk) == chunksize:
                            self._load_data_chunk(chunk)
                            chunk = []
                            n_loaded += len(chunk)
                            logger.info(
                                f"Loaded {n_loaded} spectra in {datetime.datetime.now() - beg} hours."
                            )

            if len(chunk) > 0:
                self._load_data_chunk(chunk)
        logger.info(
            f"Loaded {n_loaded} spectra in {datetime.datetime.now() - beg} hours."
        )
        return self.dgw.list_spectrum_ids()

    def _load_data_chunk(self, chunk: List[Dict[str, Any]]):
        """Load data using ProcessPool."""
        chunks = np.array_split(chunk, self.n_workers)
        with futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futs = [
                executor.submit(self._load, ch, self.n_decimals, self.min_peaks)
                for ch in chunks
            ]
        results = [f.result() for f in futs]

    @staticmethod
    def _load(chunk: List[Dict[str, Any]], n_decimals: int, min_peaks: int):
        """Clean GNPS data item and save it on Redis."""
        spectra = [{k: item[k] for k in KEYS} for item in chunk]
        data_cleaner = SpectrumProcessor()
        cleaned_data = [
            data_cleaner.process_data(spectrum_data) for spectrum_data in spectra
        ]
        cleaned_data = [spectrum for spectrum in cleaned_data if spectrum]
        spectra_data = [
            SpectrumDocumentData(spectrum, n_decimals) for spectrum in cleaned_data
        ]

        dgw = RedisSpectrumDataGateway()
        dgw.write_spectrum_documents(spectra_data)
