from __future__ import annotations

from typing import List, Optional

from drfs.filesystems.base import FileSystemBase

from omigami.gateways.fs_data_gateway import FSDataGateway


class Spec2VecFSDataGateway(FSDataGateway):
    """Data gateway for storage."""

    def __init__(
        self,
        fs: Optional[FileSystemBase] = None,
    ):
        super().__init__(fs)


class FileSystemDocumentIterator:
    """An iterator that yields document objects of files one by one to the Word2Vec model for training.
    Reading chunks is not supported by Gensim's Word2Vec model at the moment.
    """

    def __init__(self, fs_dgw: FSDataGateway, document_paths: List[str]):
        self._fs_dgw = fs_dgw
        self._document_paths = document_paths
        self._len = None

    def __iter__(self):
        self._len = 0
        for doc_path in self._document_paths:
            documents = self._fs_dgw.read_from_file(doc_path)
            for document in documents:
                self._len += 1
                yield document

        self._has_length = True

    def __len__(self):

        if not self._len:
            for doc_path in self._document_paths:
                documents = self._fs_dgw.read_from_file(doc_path)
                self._len += len(documents)

        return self._len
