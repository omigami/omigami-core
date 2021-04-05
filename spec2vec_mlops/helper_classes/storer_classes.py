from datetime import datetime
from typing import List

import pandas as pd
from feast import ValueType, Client, FeatureTable, Entity, Feature, FileSource
from feast.data_format import ParquetFormat
from matchms import Spectrum
from spec2vec import SpectrumDocument
from spec2vec_mlops.helper_classes.base_storer import BaseStorer
from spec2vec_mlops.helper_classes.embedding import Embedding

from spec2vec_mlops import config

KEYS = config["cleaned_data"]["necessary_keys"].get(list)

not_string_features2types = {
    "mz_list": ValueType.DOUBLE_LIST,
    "intensity_list": ValueType.DOUBLE_LIST,
    "losses": ValueType.DOUBLE_LIST,
    "precursor_mz": ValueType.FLOAT,
    "charge": ValueType.INT64,
    "parent_mass": ValueType.FLOAT,
}
string_features2types = {
    key.lower(): ValueType.STRING
    for key in KEYS
    if key.lower() not in not_string_features2types.keys()
}


class FeastTable:
    def __init__(
        self, out_dir: str, feast_core_url: str, feature_table_name: str, **kwargs
    ):
        self.out_dir = out_dir
        self.client = Client(core_url=feast_core_url, telemetry=False)
        self.feature_table_name = feature_table_name
        self.features2types = {**kwargs}

    def get_or_create_table(
        self, entity_description: str, entity_name="spectrum_id",
    ) -> FeatureTable:
        existing_tables = [table.name for table in self.client.list_feature_tables()]
        if self.feature_table_name in existing_tables:
            feature_table = self.client.get_feature_table(self.feature_table_name)
        else:
            feature_table = self._create_table(entity_name, entity_description)
        return feature_table

    def _create_table(self, entity_description: str, entity_name="spectrum_id",) -> FeatureTable:
        entity = Entity(
            name=entity_name,
            description=entity_description,
            value_type=ValueType.INT64,
        )
        features = [
            Feature(feature, dtype=feature_type)
            for feature, feature_type in self.features2types.items()
        ]
        batch_source = FileSource(
            file_format=ParquetFormat(),
            file_url=str(self.out_dir),
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


class SpectrumStorer(BaseStorer):
    def __init__(self, out_dir: str, feast_core_url: str, feature_table_name: str):
        self._feast_table = FeastTable(
            out_dir,
            feast_core_url,
            feature_table_name,
            **string_features2types,
            **not_string_features2types,
        )
        self._spectrum_table = self._feast_table.get_or_create_table(
            entity_description="Spectrum identifier"
        )

    def store(self, data: List[Spectrum]):
        data_df = self._get_data_df(data)
        self._feast_table.client.ingest(self._spectrum_table, data_df)

    def _get_data_df(self, data: List[Spectrum]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "spectrum_id": spectrum.metadata["spectrum_id"],
                    "mz_list": spectrum.peaks.mz,
                    "intensity_list": spectrum.peaks.intensities,
                    "losses": spectrum.losses,
                    **{
                        key: spectrum.metadata[key]
                        for key in self._feast_table.features2types.keys()
                        if key in spectrum.metadata.keys()
                    },
                    "event_timestamp": self._convert_create_time(
                        spectrum.metadata.get("create_time")
                    ),
                    "created_timestamp": datetime.now(),
                }
                for spectrum in data
            ]
        )

    @staticmethod
    def _convert_create_time(create_time: str) -> datetime:
        if create_time:
            return datetime.strptime(create_time, "%Y-%m-%d %H:%M:%S.%f")
        else:
            return datetime.now()


class DocumentStorer(BaseStorer):
    def __init__(self, out_dir: str, feast_core_url: str, feature_table_name: str):
        self._feast_table = FeastTable(
            out_dir,
            feast_core_url,
            feature_table_name,
            **{
                "words": ValueType.DOUBLE_LIST,
                "losses": ValueType.DOUBLE_LIST,
                "weights": ValueType.DOUBLE_LIST,
            },
        )
        self._document_table = self._feast_table.get_or_create_table(
            entity_description="Document identifier"
        )

    def store(self, data: List[SpectrumDocument]):
        data_df = self._get_data_df(data)
        self._feast_table.client.ingest(self._document_table, data_df)

    def _get_data_df(self, data: List[SpectrumDocument]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "spectrum_id": document.metadata["spectrum_id"],
                    "words": document.words,
                    "losses": document.losses,
                    "weights": document.weights,
                    "event_timestamp": self._convert_create_time(
                        document.metadata.get("create_time")
                    ),
                }
                for document in data
            ]
        )

    @staticmethod
    def _convert_create_time(create_time: str) -> datetime:
        if create_time:
            return datetime.strptime(create_time, "%Y-%m-%d %H:%M:%S.%f")
        else:
            return datetime.now()


class EmbeddingStorer(BaseStorer):
    def __init__(self, out_dir: str, feast_core_url: str, feature_table_name: str, run_id: str):
        self._feast_table = FeastTable(
            out_dir,
            feast_core_url,
            feature_table_name,
            **{"run_id": ValueType.STRING, "embedding": ValueType.DOUBLE_LIST},
        )
        self.run_id = run_id
        self._embedding_table = self._feast_table.get_or_create_table(
            entity_description="Embedding identifier"
        )

    def store(self, data: List[Embedding]):
        df = self._get_data_df(data)
        self._feast_table.client.ingest(self._embedding_table, df)

    def _get_data_df(self, embeddings: List[Embedding]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "spectrum_id": embedding.spectrum_id,
                    "embedding": embedding.vector,
                    "run_id": self.run_id,
                    "event_timestamp": datetime.now(),
                    "create_timestamp": datetime.now(),
                }
                for embedding in embeddings
            ]
        )
