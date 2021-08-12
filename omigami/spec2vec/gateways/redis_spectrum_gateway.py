from __future__ import annotations

import os
import pickle
from typing import List

from omigami.gateways.redis_spectrum_data_gateway import (
    RedisSpectrumDataGateway,
)

from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from spec2vec import SpectrumDocument


class Spec2VecRedisSpectrumDataGateway(RedisSpectrumDataGateway):
    """Data gateway for storage."""

    project = "spec2vec"

    def write_spectrum_documents(
        self, spectrum_data: List[SpectrumDocumentData], save_directory: str
    ):
        """Write spectra data to S3. Will overwrite a file with the same name. The spectra ids and precursor_MZ are
        required."""

        document_data = [doc.document for doc in spectrum_data]

        with open(save_directory, "wb") as f:
            pickle.dump(document_data, f)

    def list_missing_documents(
        self, spectrum_ids: List[str], documents_directory: str
    ) -> List[str]:
        """Check whether document exist.
        Return a list of IDs that do not exist.
        """

        document_file_names = os.listdir(documents_directory)
        documents = []

        for document_file in document_file_names:
            document_file_dir = f"{documents_directory}/{document_file}"
            loaded_documents = self.read_documents(document_file_dir)

            documents = documents + loaded_documents

        stored_spectrum_ids = [
            document.metadata["spectrum_id"] for document in documents
        ]

        new_spectra = set(spectrum_ids) - set(stored_spectrum_ids)
        return list(new_spectra)

    def read_documents(self, load_directory: str) -> List[SpectrumDocument]:
        """Read the document information from spectra IDs.
        Return a list of SpectrumDocument objects."""

        return pickle.load(open(load_directory, "rb"))
