from abc import ABC, abstractmethod
from typing import Dict, List


class InputDataGateway(ABC):
    @abstractmethod
    def download_gnps(self, uri: str, output_dir: str) -> str:
        pass

    @abstractmethod
    def load_gnps(self, path: str) -> List[Dict[str, str]]:
        pass
