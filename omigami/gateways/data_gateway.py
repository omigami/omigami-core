from abc import ABC, abstractmethod
from typing import List, Any

from omigami.spec2vec.entities.data_models import SpectrumInputData


class DataGateway(ABC):
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
    def read_from_file(self, path: str) -> Any:
        pass
