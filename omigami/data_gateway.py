from abc import ABC, abstractmethod
from logging import Logger
from typing import List, Iterable, Any, Set

from spec2vec import SpectrumDocument

from omigami.spec2vec.entities.data_models import SpectrumInputData
from omigami.spec2vec.entities.embedding import Embedding
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData


class InputDataGateway(ABC):
    @abstractmethod
    def download_gnps(self, uri: str, output_path: str):
        pass

    @abstractmethod
    def get_spectrum_ids(self, path: str) -> List[str]:
        pass

    @abstractmethod
    def load_spectrum(self, path: str) -> SpectrumInputData:
        pass

    @abstractmethod
    def load_spectrum_ids(
        self, path: str, spectrum_ids: List[str]
    ) -> SpectrumInputData:
        pass

    @abstractmethod
    def chunk_gnps(
        self, gnps_path: str, chunk_size: int, ion_mode: str, logger=None
    ) -> List[str]:
        pass

    @abstractmethod
    def serialize_to_file(self, path: str, object: Any) -> bool:
        pass

    @abstractmethod
    def download_ms2deepscore_model(self, uri: str, output_path: str):
        pass


class SpectrumDataGateway(ABC):
    @abstractmethod
    def write_spectrum_documents(self, spectra_data: List[SpectrumDocumentData]):
        """Write spectrum and document to Redis. Also write a sorted set of spectrum_ids."""
        pass

    @abstractmethod
    def write_embeddings(
        self, embeddings: List[Embedding], run_id: str, logger: Logger = None
    ):
        """Write embeddings to Redis."""
        pass

    @abstractmethod
    def list_existing_spectra(self, spectrum_ids: List[str]) -> Set[str]:
        """Check whether the spectrum ids exist on the database.
        Return a set of existing spectrum IDs.

        Parameters
        ----------
        spectrum_ids:
            List of spectrum ids that will be verified

        Returns
        -------
        existing_spectrum_ids:
            Subset of the input ids that exist on the database

        """
        pass

    @abstractmethod
    def list_spectrum_ids(self) -> List[str]:
        """List available spectrum IDs"""
        pass

    @abstractmethod
    def read_documents(self, spectrum_ids: List[str] = None) -> List[SpectrumDocument]:
        pass

    @abstractmethod
    def read_documents_iter(self, spectrum_ids: List[str] = None) -> Iterable:
        pass

    @abstractmethod
    def delete_spectra(self, spectrum_ids: List[str]):
        """Deletes spectra using their IDs."""
        pass
