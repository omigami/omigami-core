from abc import ABC, abstractmethod

from feast import FeatureTable


class BaseStorer(ABC):
    @abstractmethod
    def get_or_create_table(self, entity_name: str, entity_description: str) -> FeatureTable:
        pass

    @abstractmethod
    def _create_table(self, entity_name: str, entity_description: str) -> FeatureTable:
        pass
