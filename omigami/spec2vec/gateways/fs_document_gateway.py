from __future__ import annotations

from typing import List, Optional

from drfs.filesystems.base import FileSystemBase
from spec2vec import SpectrumDocument

from omigami.gateways import RedisSpectrumDataGateway
from omigami.gateways.fs_data_gateway import FSDataGateway


# TODO: Here I'm not sure what you would prefer. Either we can pass the RedisDGW like it is or we could pass it with
#  every function call or we could do a double inheritance. Lemme know.
class Spec2VecFSDataGateway(FSDataGateway):
    """Data gateway for storage."""

    def __init__(
        self,
        redis_dgw: RedisSpectrumDataGateway,
        ion_mode: str = None,
        fs: Optional[FileSystemBase] = None,
    ):
        super().__init__(fs)
        self._redis_dgw = redis_dgw
        self._ion_mode = ion_mode

    def serialize_documents(self, path: str, documents: List[SpectrumDocument]):

        for doc in documents:
            self._redis_dgw.write_document_id(
                document_id=doc.get("spectrum_id"),
                ion_mode=self._ion_mode,
            )

        self.serialize_to_file(path, documents)

    def list_missing_documents(self, document_ids: List[str]) -> List[str]:
        """Check whether document ids exist in redis.
        Return a list of IDs that do not exist.
        """

        stored_document_ids = self._redis_dgw.list_document_ids(ion_mode=self._ion_mode)
        stored_document_ids = [ID.decode("utf-8") for ID in stored_document_ids]
        new_spectra = set(document_ids) - set(stored_document_ids)
        return list(new_spectra)

    def read_documents_iter(self, document_paths):
        return FileSystemDocumentIterator(self, document_paths)


class FileSystemDocumentIterator:
    """An iterator that yields document objects of files one by one to the Word2Vec model for training.
    Reading chunks is not supported by Gensim's Word2Vec model at the moment.
    """

    def __init__(
        self,
        dgw: Spec2VecFSDataGateway,
        document_paths: List[str] = None,
    ):
        self._dgw = dgw
        self._document_paths = document_paths
        self._has_length = False
        self._len = 0

    def __iter__(self):

        for doc_path in self._document_paths:
            documents = self._dgw.read_from_file(doc_path)
            for document in documents:
                if not self._has_length:
                    self._len += 1
                yield document

        self._has_length = True

    def __len__(self):

        if not self._has_length:
            for doc in self:
                pass

        return self._len
