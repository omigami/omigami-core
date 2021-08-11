from __future__ import annotations

import os
import pickle
from typing import List, Dict

from omigami.gateways.redis_spectrum_data_gateway import (
    RedisSpectrumDataGateway,
    RedisHashesIterator,
)
from omigami.spec2vec.config import (
    DOCUMENT_HASHES,
)
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from spec2vec import SpectrumDocument


class Spec2VecRedisSpectrumDataGateway(RedisSpectrumDataGateway):
    """Data gateway for Redis storage."""

    project = "spec2vec"

    def write_spectrum_documents(
        self, spectrum_data: List[SpectrumDocumentData], save_dir: str
    ):
        """Write spectra data on the redis database. The spectra ids and precursor_MZ are required."""

        document_data = [doc.document for doc in spectrum_data]

        with open(save_dir, "wb") as f:
            pickle.dump(document_data, f)

    def list_missing_documents(
        self, spectrum_ids: List[str], documents_directory
    ) -> List[str]:
        """Check whether document exist.
        Return a list of IDs that do not exist.
        """

        if not os.path.exists(documents_directory):
            return spectrum_ids
        document_file_names = os.listdir(documents_directory)
        documents = []

        for document_file in document_file_names:
            documents = documents + self.read_documents(
                f"{documents_directory}/{document_file}"
            )

        stored_spectrum_ids = [
            document.metadata["spectrum_id"] for document in documents
        ]

        new_spectra = set(spectrum_ids) - set(stored_spectrum_ids)
        return new_spectra

    def read_documents(self, load_dir: str) -> Dict[SpectrumDocument]:
        """Read the document information from spectra IDs.
        Return a list of SpectrumDocument objects."""

        return pickle.load(open(load_dir, "rb"))
