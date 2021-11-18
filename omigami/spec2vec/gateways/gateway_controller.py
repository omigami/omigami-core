from typing import List, Optional

from spec2vec import SpectrumDocument

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)


class DocumentDataGateway:
    # TODO: see TODO on redis_spectrum_gateway.py
    def __init__(
        self,
        ion_mode: str,
        document_dgw: Spec2VecRedisSpectrumDataGateway = None,
        fs_dgw: FSDataGateway = None,
    ):
        self._ion_mode = ion_mode
        self._document_dgw = document_dgw or Spec2VecRedisSpectrumDataGateway()
        self._fs_dgw = fs_dgw or FSDataGateway()

    def write_documents(self, path: str, documents: List[Optional[SpectrumDocument]]):
        self._document_dgw.write_document_ids(documents, self._ion_mode)
        self._fs_dgw.serialize_to_file(path, documents)

    def remove_documents_file(self, path: str):
        documents = self._fs_dgw.read_from_file(path)
        document_ids = [doc.get("spectrum_id") for doc in documents]
        self._document_dgw.remove_document_ids(document_ids, self._ion_mode)
        self._fs_dgw.remove_file(path)
