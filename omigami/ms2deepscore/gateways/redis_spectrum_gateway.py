from __future__ import annotations

import pickle
from logging import Logger
from typing import List
from spec2vec import SpectrumDocument

from omigami.ms2deepscore.gateways.redis_spectrum_data_gateway import (
    RedisSpectrumDataGateway,
    RedisHashesIterator,
)
from omigami.spec2vec.config import (
    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
    DOCUMENT_HASHES,
    EMBEDDING_HASHES,
)
from omigami.spec2vec.entities.embedding import Embedding
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData


class MS2DeepScoreRedisSpectrumDataGateway(RedisSpectrumDataGateway):
    """Data gateway for Redis storage."""

    def write_spectrum_documents(self, spectrum_data: List[SpectrumDocumentData]):
        """Write spectra data on the redis database. The spectra ids and precursor_MZ are required."""
        self._init_client()
        pipe = self.client.pipeline()
        for spectrum in spectrum_data:
            spectrum_info = spectrum.spectrum
            document = spectrum.document
            if spectrum_info and document:
                pipe.zadd(
                    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
                    {spectrum.spectrum_id: spectrum.precursor_mz},
                )

                pipe.hset(DOCUMENT_HASHES, spectrum.spectrum_id, pickle.dumps(document))
        pipe.execute()

    def write_embeddings(
        self, embeddings: List[Embedding], run_id: str, logger: Logger = None
    ):
        """Write embeddings data on the redis database."""
        self._init_client()
        if logger:
            logger.debug(
                f"Saving {len(embeddings)} embeddings to the client {self.client}"
                f" on hash '{EMBEDDING_HASHES}_{run_id}'."
            )
        pipe = self.client.pipeline()
        for embedding in embeddings:
            pipe.hset(
                f"{EMBEDDING_HASHES}_{run_id}",
                embedding.spectrum_id,
                pickle.dumps(embedding),
            )
        pipe.execute()

    # Not used atm
    def list_documents_not_exist(self, spectrum_ids: List[str]) -> List[str]:
        """Check whether document exist on Redis.
        Return a list of IDs that do not exist.
        """
        self._init_client()
        return self._list_spectrum_ids_not_exist(DOCUMENT_HASHES, spectrum_ids)

    def read_documents(self, spectrum_ids: List[str] = None) -> List[SpectrumDocument]:
        """Read the document information from spectra IDs.
        Return a list of SpectrumDocument objects."""
        self._init_client()
        return self._read_hashes(DOCUMENT_HASHES, spectrum_ids)

    def read_embeddings(
        self, run_id: str, spectrum_ids: List[str] = None
    ) -> List[Embedding]:
        """Read the embeddings from spectra IDs.
        Return a list of Embedding objects."""
        self._init_client()
        return self._read_hashes(f"{EMBEDDING_HASHES}_{run_id}", spectrum_ids)

    def read_documents_iter(
        self, spectrum_ids: List[str] = None
    ) -> RedisHashesIterator:
        """Returns an iterator that yields Redis object one by one"""
        self._init_client()
        return RedisHashesIterator(self, DOCUMENT_HASHES, spectrum_ids)

    def _list_spectrum_ids_not_exist(
        self, hash_name: str, spectrum_ids: List[str]
    ) -> List[str]:
        self._init_client()
        return [id for id in spectrum_ids if not self.client.hexists(hash_name, id)]
