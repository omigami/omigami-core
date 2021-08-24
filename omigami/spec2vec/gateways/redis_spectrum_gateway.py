from typing import Set, List, Optional

from spec2vec import SpectrumDocument

from omigami.gateways import RedisSpectrumDataGateway
from omigami.spec2vec.config import PROJECT_NAME, DOCUMENT_HASHES
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData


class Spec2VecRedisSpectrumDataGateway(RedisSpectrumDataGateway):
    def __init__(self, project=PROJECT_NAME):

        super().__init__(project)

    def list_document_ids(self, ion_mode: str) -> Set[str]:
        self._init_client()

        key_name = self._format_redis_key(hashes=DOCUMENT_HASHES, ion_mode=ion_mode)

        return self.client.smembers(name=key_name)

    def write_document_ids(
        self, document_ids: List[Optional[SpectrumDocument]], ion_mode: str
    ):
        self._init_client()

        key_name = self._format_redis_key(hashes=DOCUMENT_HASHES, ion_mode=ion_mode)

        for doc in document_ids:
            self.client.sadd(
                key_name,
                doc.get("spectrum_id"),
            )

    def remove_document_ids(self, document_ids: List[str], ion_mode: str):
        """Only needed for testing. Removes a document_id from redis"""
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
        stored_document_ids = [
            document_id.decode("utf-8") for document_id in stored_document_ids
        ]
        new_spectra = set(document_ids) - set(stored_document_ids)
        return list(new_spectra)
