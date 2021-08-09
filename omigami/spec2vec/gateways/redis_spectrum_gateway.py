from __future__ import annotations

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

    def write_spectrum_documents(self, spectrum_data: List[SpectrumDocumentData]):
        """Write spectra data on the redis database. The spectra ids and precursor_MZ are required."""
        self._init_client()
        pipe = self.client.pipeline()
        for spectrum in spectrum_data:
            spectrum_info = spectrum.spectrum
            document = spectrum.document
            if document:
                pipe.hset(DOCUMENT_HASHES, spectrum.spectrum_id, pickle.dumps(document))
        pipe.execute()

    def list_missing_documents(self, spectrum_ids: List[str]) -> List[str]:
        """Check whether document exist on Redis.
        Return a list of IDs that do not exist.
        """
        self._init_client()
        return self._list_missing_spectrum_ids(DOCUMENT_HASHES, spectrum_ids)

    def read_documents(self, spectrum_ids: List[str] = None) -> List[SpectrumDocument]:
        """Read the document information from spectra IDs.
        Return a list of SpectrumDocument objects."""
        self._init_client()
        return self._read_hashes(DOCUMENT_HASHES, spectrum_ids)

    def read_documents_iter(
        self, spectrum_ids: List[str] = None
    ) -> RedisHashesIterator:
        """Returns an iterator that yields Redis object one by one"""
        self._init_client()
        return RedisHashesIterator(self, DOCUMENT_HASHES, spectrum_ids)
