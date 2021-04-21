import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

import pandas as pd
from feast import ValueType, Client, FeatureTable, Entity, Feature, FileSource
from feast.data_format import ParquetFormat
from spec2vec_mlops import config

from spec2vec_mlops.helper_classes.exception import StorerLoadError
from spec2vec_mlops.helper_classes.feast_utils import FeastUtils

FEAST_JOB_SERVICE_URL = os.getenv("FEAST_JOB_SERVICE_URL", None)
FEAST_CORE_URL = os.getenv(
    "FEAST_CORE_URL",
    config["feast"]["url"]["local"],
)
FEAST_SERVING_URL = os.getenv(
    "FEAST_SERVING_URL",
    config["feast"]["serving_url"]["local"],
)
FEAST_BASE_SOURCE_LOCATION = os.getenv(
    "FEAST_BASE_SOURCE_LOCATION",
    config["feast"]["spark"]["base_source_location"],
)
FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION = os.getenv(
    "FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION",
    config["feast"]["spark"]["output_location"],
)
FEAST_HISTORICAL_FEATURE_OUTPUT_READ_LOCATION = os.getenv(
    "FEAST_HISTORICAL_FEATURE_OUTPUT_READ_LOCATION",
    config["feast"]["spark"]["output_location"],
)


class FeastDataGateway:
    """
    Data gateway for Feast storage.

    This class is independent of the business entities.
    """

    def __init__(self):
        if FEAST_JOB_SERVICE_URL:
            self.client = Client(
                core_url=FEAST_CORE_URL,
                serving_url=FEAST_SERVING_URL,
                job_service_url=FEAST_JOB_SERVICE_URL,
                telemetry=False,
            )
        else:
            self.client = Client(
                core_url=FEAST_CORE_URL,
                serving_url=FEAST_SERVING_URL,
                telemetry=False,
            )

    def store_offline(self, table: FeatureTable, data_df: pd.DataFrame):
        """Write data to offline store."""
        self.client.ingest(table, data_df)

    def store_online(self, table: FeatureTable, all=True):
        """Write data from offline to online store.

        Parameters
        ----------
        table: FeatureTable
            Feature table to be stored online
        all: bool
            If True, write all values.
            If False, write only data with event_timestamp from yesterday to today.
        """
        end = datetime.now()
        start = datetime.fromtimestamp(0) if all else end - timedelta(1)
        job = self.client.start_offline_to_online_ingestion(table, start, end)
        FeastUtils.wait_for_job(job)

    def read_offline(
        self, feature_list: List[str], entity_dict: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """Read data from offline store.

        entity_dict must be in the following format:
        {
            "spectrum_id": ["AA", "BB"],
            "event_timestamp": [datetime.now(), datetime.now()]
        }
        """
        job = self.client.get_historical_features(
            feature_list,
            entity_source=pd.DataFrame(entity_dict),
            output_location=FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION,
        )
        FeastUtils.wait_for_job(job)
        return FeastUtils.read_parquet(FEAST_HISTORICAL_FEATURE_OUTPUT_READ_LOCATION)

    def read_online(
        self, feature_list: List[str], entity_rows: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Read data from online store.

        entity_rows must be in the following format:
        [{"spectrum_id": "AA"}, {"spectrum_id": "BB"}]]
        """
        responses = []
        for feature in feature_list:
            response = self.client.get_online_features(
                feature_refs=[feature],
                entity_rows=entity_rows,
            )
            df = pd.DataFrame(response.to_dict())
            indices = list(entity_rows[0].keys())
            df = df.set_index(indices, drop=True)
            responses.append(df)
        return pd.concat(responses, axis=1)

    def get_or_create_table(
        self,
        entity_description: str,
        entity_name: str,
        feature_table_name: str,
        features2types: Dict[str, ValueType],
    ) -> FeatureTable:
        existing_tables = [table.name for table in self.client.list_feature_tables()]
        if feature_table_name in existing_tables:
            feature_table = self.client.get_feature_table(feature_table_name)
        else:
            feature_table = self._create_table(
                entity_description, entity_name, feature_table_name, features2types
            )
        return feature_table

    def _create_table(
        self,
        entity_description: str,
        entity_name: str,
        feature_table_name: str,
        features2types: Dict[str, ValueType],
    ) -> FeatureTable:
        """Create feature table."""
        entity = Entity(
            name=entity_name,
            description=entity_description,
            value_type=ValueType.STRING,
        )
        features = [
            Feature(feature, dtype=feature_type)
            for feature, feature_type in features2types.items()
        ]
        batch_source = FileSource(
            file_format=ParquetFormat(),
            file_url=str(os.path.join(FEAST_BASE_SOURCE_LOCATION, feature_table_name)),
            event_timestamp_column="event_timestamp",
            created_timestamp_column="created_timestamp",
        )
        feature_table = FeatureTable(
            name=feature_table_name,
            entities=[entity_name],
            features=features,
            batch_source=batch_source,
        )
        self.client.apply(entity)
        self.client.apply(feature_table)
        return feature_table
