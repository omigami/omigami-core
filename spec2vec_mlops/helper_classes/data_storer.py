import datetime
import os
from dataclasses import dataclass
from typing import List

import pandas as pd
from feast import ValueType, Client, FeatureTable, Entity, Feature, FileSource
from feast.data_format import ParquetFormat
from matchms import Spectrum
from spec2vec import SpectrumDocument

from spec2vec_mlops import config

KEYS = config["cleaned_data"]["necessary_keys"].get(list)


@dataclass
class SpectrumData:
    spectrum_info: FeatureTable
    spectrum_meta: FeatureTable


class DataStorer:
    def __init__(self, out_dir: str, feast_core_url: str):
        self.client = Client(core_url=feast_core_url, telemetry=False)
        self.feature_entity_name = "spectrum_id"
        self.meta_entity_name = "spectrum_meta_id"
        self.feature_table_name = "spectrum_info"
        self.meta_table_name = "spectrum_meta"
        self.out_data_dir = os.path.join(out_dir, "data")
        self.out_meta_dir = os.path.join(out_dir, "meta")

        not_string_features2types = {
            "mz_list": ValueType.DOUBLE_LIST,
            "intensity_list": ValueType.DOUBLE_LIST,
            "precursor_mz": ValueType.FLOAT,
            "charge": ValueType.INT64,
            "parent_mass": ValueType.FLOAT,
            "words": ValueType.DOUBLE_LIST,
            "losses": ValueType.DOUBLE_LIST,
            "weights": ValueType.DOUBLE_LIST,
        }
        string_features2types = {
            key.lower(): ValueType.STRING
            for key in KEYS
            if key.lower() not in not_string_features2types.keys()
        }
        self.features2types = {**not_string_features2types, **string_features2types}
        self.spectrum_data = self._get_or_create_spectrum_table()

    def _get_or_create_spectrum_table(self) -> SpectrumData:
        entity_names = [e.name for e in self.client.list_entities()]
        if self.feature_entity_name not in entity_names:
            self._create_spectrum_entity()

        if self.meta_entity_name not in entity_names:
            self._create_meta_entity()

        table_names = [t.name for t in self.client.list_feature_tables()]
        if self.feature_table_name not in table_names:
            spectrum_info = self._create_spectrum_info_table()
        else:
            spectrum_info = self.client.get_feature_table(self.feature_table_name)

        if self.meta_table_name not in table_names:
            spectrum_meta = self._create_spectrum_meta_table()
        else:
            spectrum_meta = self.client.get_feature_table(self.meta_table_name)

        spectrum_data = SpectrumData(
            spectrum_info=spectrum_info, spectrum_meta=spectrum_meta
        )
        return spectrum_data

    def _create_spectrum_entity(self) -> Entity:
        spectrum_id = Entity(
            name=self.feature_entity_name,
            description="Spectrum identifier",
            value_type=ValueType.STRING,
        )
        self.client.apply(spectrum_id)
        return spectrum_id

    def _create_meta_entity(self) -> Entity:
        spectrum_meta_id = Entity(
            name=self.meta_entity_name,
            description="Spectrum metadata identifier",
            value_type=ValueType.INT64,
        )
        self.client.apply(spectrum_meta_id)
        return spectrum_meta_id

    def _create_spectrum_info_table(self) -> FeatureTable:
        features = [
            Feature(feature, dtype=feature_type)
            for feature, feature_type in self.features2types.items()
        ]
        batch_source = FileSource(
            file_format=ParquetFormat(),
            file_url=str(self.out_data_dir),
            event_timestamp_column="event_timestamp",
            created_timestamp_column="created_timestamp",
        )
        spectrum_info = FeatureTable(
            name=self.feature_table_name,
            entities=[self.feature_entity_name],
            features=features,
            batch_source=batch_source,
        )

        self.client.apply(spectrum_info)
        return spectrum_info

    def _create_spectrum_meta_table(self) -> FeatureTable:
        meta_source = FileSource(
            file_format=ParquetFormat(),
            file_url=str(self.out_meta_dir),
            event_timestamp_column="event_timestamp",
            created_timestamp_column="created_timestamp",
        )
        all_spectrum_ids = Feature("all_spectrum_ids", dtype=ValueType.STRING_LIST)
        spectrum_meta = FeatureTable(
            name=self.meta_table_name,
            entities=[self.meta_entity_name],
            features=[all_spectrum_ids],
            batch_source=meta_source,
        )

        self.client.apply(spectrum_meta)
        return spectrum_meta

    def store_cleaned_data(self, data: List[Spectrum]):
        data_df = self._get_cleaned_data_df(data)
        self.client.ingest(self.spectrum_data.spectrum_info, data_df)

        # TODO: add existing ids
        all_spectrum_ids = data_df["spectrum_id"].tolist()
        meta_df = pd.DataFrame.from_records(
            [
                {
                    self.meta_entity_name: 1,
                    "all_spectrum_ids": all_spectrum_ids,
                    "event_timestamp": datetime.datetime.now(),
                    "created_timestamp": datetime.datetime.now(),
                }
            ]
        )
        self.client.ingest(self.spectrum_data.spectrum_meta, meta_df)

    def _get_cleaned_data_df(self, data: List[Spectrum]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    self.feature_entity_name: spectrum.metadata["spectrum_id"],
                    "mz_list": spectrum.peaks.mz,
                    "intensity_list": spectrum.peaks.intensities,
                    "words": [],
                    "losses": [],
                    "weights": [],
                    **{
                        key: spectrum.metadata[key]
                        for key in self.features2types.keys()
                        if key in spectrum.metadata.keys()
                    },
                    "event_timestamp": datetime.datetime.now(),
                    "created_timestamp": datetime.datetime.now(),
                }
                for spectrum in data
            ]
        )

    def store_documents(self, data: List[SpectrumDocument]):
        data_df = self._get_documents_df(data)
        self.client.ingest(self.spectrum_data.spectrum_info, data_df)

        # TODO: add existing ids
        all_spectrum_ids = data_df["spectrum_id"].tolist()
        meta_df = pd.DataFrame.from_records(
            [
                {
                    self.meta_entity_name: 1,
                    "all_spectrum_ids": all_spectrum_ids,
                    "event_timestamp": datetime.datetime.now(),
                    "created_timestamp": datetime.datetime.now(),
                }
            ]
        )
        self.client.ingest(self.spectrum_data.spectrum_meta, meta_df)

    def _get_documents_df(self, data: List[SpectrumDocument]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    self.feature_entity_name: document.metadata["spectrum_id"],
                    "words": document.words,
                    "losses": document.losses,
                    "weights": document.weights,
                    "event_timestamp": datetime.datetime.now(),
                    "created_timestamp": datetime.datetime.now(),
                }
                for document in data
            ]
        )
