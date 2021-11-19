from abc import ABC, abstractmethod
from typing import List, Optional

from spec2vec import SpectrumDocument


class SpectrumDocumentDataGateway(ABC):
    @abstractmethod
    def list_document_ids(self, ion_mode: str) -> List[str]:
        """List documents IDs for a given ion mode

        Parameters
        ----------
        ion_mode:
            Document ion mode.

        Returns
        -------
        List of IDs available
        """
        pass

    @abstractmethod
    def write_documents(
        self, documents: List[Optional[SpectrumDocument]], ion_mode: str
    ):
        """Saves the documents to the storage system

        Parameters
        ----------
        documents:
            Documents to be saved
        ion_mode:
            Document ion mode.

        """
        pass

    @abstractmethod
    def remove_documents(self, document_ids: List[str], ion_mode: str):
        """Deletes documents from the storage using their IDs

        Parameters
        ----------
        document_ids:
            List of document IDs to be deleted
        ion_mode:
            Documents ion mode

        """
        pass

    @abstractmethod
    def list_missing_documents(
        self, document_ids: List[str], ion_mode: str
    ) -> List[str]:
        """Given a list of document IDs, return the ones that don't exist in the
        database.

        Parameters
        ----------
        document_ids:
            List of IDs to check
        ion_mode:
            Documents ion mode

        Returns
        -------
        List of IDs that are not present in the storage system
        """
        pass
