from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from feast import ValueType, Client, FeatureTable, Entity, Feature, FileSource
from feast.data_format import ParquetFormat
from matchms import Spectrum
from spec2vec import SpectrumDocument

from spec2vec_mlops import config

KEYS = config["cleaned_data"]["necessary_keys"].get(list)


class DataStorer:
    def __init__(self, out_dir: str, feast_core_url: str):
        self.client = Client(core_url=feast_core_url, telemetry=False)
        self.feature_table_name = "spectrum_info"
        self.out_dir = out_dir

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
        self.spectrum_info = self._get_or_create_spectrum_table()

    def _get_or_create_spectrum_table(self) -> FeatureTable:
        if not any(
            table.name != self.feature_table_name
            for table in self.client.list_feature_tables()
        ):
            spectrum_info = self._create_spectrum_info_table()
        else:
            spectrum_info = self.client.get_feature_table(self.feature_table_name)
        return spectrum_info

    def _create_spectrum_info_table(self) -> FeatureTable:
        spectrum_id = Entity(
            name="spectrum_id",
            description="Spectrum identifier",
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
        spectrum_info = FeatureTable(
            name=self.feature_table_name,
            entities=["spectrum_id"],
            features=features,
            batch_source=batch_source,
        )
        self.client.apply(spectrum_id)
        self.client.apply(spectrum_info)
        return spectrum_info

    def store_cleaned_data(self, data: List[Spectrum]):
        data_df = self._get_cleaned_data_df(data)
        self.client.ingest(self.spectrum_info, data_df)

    def _get_cleaned_data_df(self, data: List[Spectrum]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "spectrum_id": spectrum.metadata["spectrum_id"],
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
                    "event_timestamp": spectrum.metadata.get("create_time", datetime.now()),
                    "created_timestamp": datetime.now(),
                }
                for spectrum in data
            ]
        )

    def store_documents(self, data: List[SpectrumDocument]):
        data_df = self._get_documents_df(data)
        self.client.ingest(self.spectrum_info, data_df)

    def _get_documents_df(self, data: List[SpectrumDocument]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "spectrum_id": document.metadata["spectrum_id"],
                    "words": document.words,
                    "losses": document.losses,
                    "weights": document.weights,
                    "event_timestamp": document.metadata.get("create_time", datetime.now()),
                }
                for document in data
            ]
        )

    def store_embeddings(self, data: List[SpectrumDocument], embeddings: List[np.ndarray]):
        df = self._get_embeddings_df(data, embeddings)
        self.client.ingest(self.spectrum_info, df)

    def _get_embeddings_df(self, data: List[SpectrumDocument], embeddings: List[np.ndarray]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "spectrum_id": document.metadata["spectrum_id"],
                    "embeddings": embedding,
                    "event_timestamp": document.metadata.get("create_time", datetime.now())
                }
                for document, embedding in zip(data, embeddings)
            ]
        )
