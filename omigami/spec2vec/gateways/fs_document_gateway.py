from __future__ import annotations

import os
from typing import List

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData


class Spec2VecFSDataGateway(FSDataGateway):
    """Data gateway for storage."""

    def list_missing_documents(
        self, spectrum_ids: List[str], documents_directory: str
    ) -> List[str]:
        """Check whether documents exist in the given path on the declared filesystem.
        Return a list of IDs that do not exist.
        """

        document_file_names = os.listdir(documents_directory)
        documents = []

        for document_file in document_file_names:
            document_file_dir = f"{documents_directory}/{document_file}"
            loaded_documents = self.read_from_file(document_file_dir)

            documents = documents + loaded_documents

        stored_spectrum_ids = [
            document.metadata["spectrum_id"] for document in documents
        ]

        new_spectra = set(spectrum_ids) - set(stored_spectrum_ids)
        return list(new_spectra)

    def read_documents_iter(self, document_paths):
        return FileSystemDocumentIterator(self, document_paths)


class FileSystemDocumentIterator:
    """An iterator that yields Redis object one by one to the word2vec model for training.
    Reading chunks is not supported by gensim word2vec at the moment.
    """

    def __init__(
        self,
        dgw: Spec2VecFSDataGateway,
        document_paths: List[str] = None,
    ):
        self._dgw = dgw
        self._document_paths = document_paths

    def __iter__(self):

        for doc_path in self._document_paths:
            documents = self._dgw.read_from_file(doc_path)
            for document in documents:
                yield document
