from abc import ABC, abstractmethod
from typing import List, Iterable, Any, Set

from spec2vec import SpectrumDocument

from omigami.entities.data_models import SpectrumInputData
from omigami.entities.embedding import Embedding
from omigami.entities.spectrum_document import SpectrumDocumentData


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
    def chunk_gnps(self, gnps_path: str, chunk_size: int) -> List[str]:
        pass

    @abstractmethod
    def serialize_to_file(self, path: str, object: Any) -> bool:
        pass


class SpectrumDataGateway(ABC):
    @abstractmethod
    def write_spectrum_documents(self, spectra_data: List[SpectrumDocumentData]):
        """Write spectrum and document to Redis. Also write a sorted set of spectrum_ids."""
        pass

    @abstractmethod
    def write_embeddings(self, embeddings: List[Embedding], run_id: str):
        """Write embeddings to Redis."""
        pass

    @abstractmethod
    def list_spectra_not_exist(self, spectrum_ids: List[str]) -> Set[str]:
        """Check whether spectra exist.
        Return a list of IDs that do not exist.
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
    def read_documents_iter(self, spectrum_ids: List[str]) -> Iterable:
        pass

    @abstractmethod
    def delete_spectra(self, spectrum_ids: List[str]):
        """Deletes spectra using their IDs."""
        pass
