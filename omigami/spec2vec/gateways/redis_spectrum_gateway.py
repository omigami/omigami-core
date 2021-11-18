from typing import List, Optional

from spec2vec import SpectrumDocument

from omigami.gateways import RedisSpectrumDataGateway
from omigami.spec2vec.config import PROJECT_NAME, DOCUMENT_HASHES


class Spec2VecRedisSpectrumDataGateway(RedisSpectrumDataGateway):
    # TODO: refactor this inheritance. It doesn't make sense given what the two classes
    # TODO: are doing at the moment. Add new documents data gateway and maybe unify with
    # TODO: the other document dgw on gateway_controller.py
    def __init__(self, project=PROJECT_NAME):
        super().__init__(project)

    def list_document_ids(self, ion_mode: str) -> List[str]:
        self._init_client()

        key_name = self._format_redis_key(hashes=DOCUMENT_HASHES, ion_mode=ion_mode)
        stored_document_ids = self.client.smembers(name=key_name)
        stored_document_ids = [
            document_id.decode("utf-8") for document_id in stored_document_ids
        ]
        return stored_document_ids

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
