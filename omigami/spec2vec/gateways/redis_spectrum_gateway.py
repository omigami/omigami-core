from __future__ import annotations

import os
import pickle
from typing import List

from omigami.gateways.redis_spectrum_data_gateway import (
    RedisSpectrumDataGateway,
    RedisHashesIterator,
)
from omigami.spec2vec.config import (
    DOCUMENT_HASHES,
)
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from spec2vec import SpectrumDocument


class Spec2VecRedisSpectrumDataGateway(RedisSpectrumDataGateway):
    """Data gateway for Redis storage."""

    project = "spec2vec"

    def write_spectrum_documents(
        self, spectrum_data: List[SpectrumDocumentData], save_dir: str
    ):
        """Write spectra data on the redis database. The spectra ids and precursor_MZ are required."""

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_file_dir = f"{save_dir}/documents.pckl"

        if os.path.isfile(save_file_dir):
            spectrum_data = self.read_documents(save_dir) + spectrum_data

        pickle.dump(spectrum_data, open(save_file_dir, "wb"))

    def list_missing_documents(self, spectrum_ids: List[str]) -> List[str]:
        """Check whether document exist on Redis.
        Return a list of IDs that do not exist.
        """
        self._init_client()
        return self._list_missing_spectrum_ids(DOCUMENT_HASHES, spectrum_ids)

    def read_documents(self, load_dir: str) -> List[SpectrumDocument]:
        """Read the document information from spectra IDs.
        Return a list of SpectrumDocument objects."""

        return pickle.load(open(f"{load_dir}/documents.pckl", "rb"))

    def read_documents_iter(
        self, spectrum_ids: List[str] = None
    ) -> RedisHashesIterator:
        """Returns an iterator that yields Redis object one by one"""
        self._init_client()
        return RedisHashesIterator(self, DOCUMENT_HASHES, spectrum_ids)
