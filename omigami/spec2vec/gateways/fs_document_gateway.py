from __future__ import annotations

import os
from typing import List

from omigami.gateways.fs_data_gateway import FSDataGateway

from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData


class Spec2VecFSDocumentDataGateway(FSDataGateway):
    """Data gateway for storage."""

    project = "spec2vec"

    def serialize_spectrum_documents(
        self, save_directory: str, spectrum_data: List[SpectrumDocumentData]
    ):
        """Write spectra data to a filesystem. Will overwrite a file with the same name. The spectra ids and
        precursor_MZ are required."""

        document_data = [doc.document for doc in spectrum_data]
        self.serialize_to_file(save_directory, document_data)

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
