import os

from feast import ValueType, Client, FeatureTable, Entity, Feature, FileSource
from feast.data_format import ParquetFormat

from spec2vec_mlops import config

FEAST_CORE_URL = os.getenv(
    "FEAST_CORE_URL",
    config["feast"]["url"]["local"],
)
FEAST_BASE_SOURCE_LOCATION = os.getenv(
    "FEAST_BASE_SOURCE_LOCATION",
    config["feast"]["spark"]["base_source_location"],
)


class FeastTableGenerator:
    def __init__(self, feature_table_name: str, **kwargs):
        self.client = Client(core_url=FEAST_CORE_URL, telemetry=False)
        self.feature_table_name = feature_table_name
        self.features2types = {**kwargs}

    def get_or_create_table(
        self,
        entity_description: str,
        entity_name="spectrum_id",
    ) -> FeatureTable:
        existing_tables = [table.name for table in self.client.list_feature_tables()]
        if self.feature_table_name in existing_tables:
            feature_table = self.client.get_feature_table(self.feature_table_name)
        else:
            feature_table = self._create_table(entity_description, entity_name)
        return feature_table

    def _create_table(
        self,
        entity_description: str,
        entity_name="spectrum_id",
    ) -> FeatureTable:
        entity = Entity(
            name=entity_name,
            description=entity_description,
            value_type=ValueType.STRING,
        )
        features = [
            Feature(feature, dtype=feature_type)
            for feature, feature_type in self.features2types.items()
        ]
        batch_source = FileSource(
            file_format=ParquetFormat(),
            file_url=str(
                os.path.join(FEAST_BASE_SOURCE_LOCATION, self.feature_table_name)
            ),
            event_timestamp_column="event_timestamp",
            created_timestamp_column="created_timestamp",
        )
        feature_table = FeatureTable(
            name=self.feature_table_name,
            entities=[entity_name],
            features=features,
            batch_source=batch_source,
        )
        self.client.apply(entity)
        self.client.apply(feature_table)
        return feature_table
