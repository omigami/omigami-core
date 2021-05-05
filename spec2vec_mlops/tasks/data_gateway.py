from abc import ABC, abstractmethod
from typing import Dict, List, Iterable

from spec2vec import SpectrumDocument

from spec2vec_mlops.entities.embedding import Embedding
from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData


class InputDataGateway(ABC):
    @abstractmethod
    def download_gnps(self, uri: str, dataset_dir: str, dataset_id: str) -> str:
        pass

    @abstractmethod
    def load_gnps(self, path: str) -> List[Dict[str, str]]:
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
    def list_spectra_not_exist(self, spectrum_ids: List[str]):
        """Check whether spectra exist on Redis.
        Return a list of IDs that do not exist.
        """
        pass

    @abstractmethod
    def read_documents(self, spectrum_ids: List[str] = None) -> List[SpectrumDocument]:
        pass

    @abstractmethod
    def read_documents_iter(self) -> Iterable:
        pass
