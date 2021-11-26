from typing import List, Optional

from spec2vec import SpectrumDocument

from omigami.spectra_matching.spec2vec.config import PROJECT_NAME, DOCUMENT_HASHES
from omigami.spectra_matching.spec2vec.storage.spectrum_document import (
    SpectrumDocumentDataGateway,
)
from omigami.spectra_matching.storage import RedisDataGateway


class RedisSpectrumDocumentDataGateway(RedisDataGateway, SpectrumDocumentDataGateway):
    def __init__(self, project: str = PROJECT_NAME):
        super().__init__(project)

    def list_document_ids(self, ion_mode: str) -> List[str]:
        self._init_client()

        key_name = self._format_redis_key(hashes=DOCUMENT_HASHES, ion_mode=ion_mode)
        stored_document_ids = self.client.smembers(name=key_name)
        stored_document_ids = [
            document_id.decode("utf-8") for document_id in stored_document_ids
        ]
        return stored_document_ids

    def write_documents(
        self, documents: List[Optional[SpectrumDocument]], ion_mode: str
    ):
        self._init_client()

        key_name = self._format_redis_key(hashes=DOCUMENT_HASHES, ion_mode=ion_mode)

        for doc in documents:
            self.client.sadd(
                key_name,
                doc.get("spectrum_id"),
            )

    def remove_documents(self, document_ids: List[str], ion_mode: str):
        self._init_client()

        key_name = self._format_redis_key(hashes=DOCUMENT_HASHES, ion_mode=ion_mode)

        for doc_id in document_ids:
            self.client.srem(key_name, doc_id)

    def list_missing_documents(
        self, document_ids: List[str], ion_mode: str
    ) -> List[str]:
        """Check whether document ids exist in redis.
        Return a list of IDs that do not exist.
        """

        stored_document_ids = self.list_document_ids(ion_mode=ion_mode)

        new_spectra = set(document_ids) - set(stored_document_ids)
        return list(new_spectra)
