from datetime import datetime
from typing import List

import pandas as pd
from feast import ValueType
from matchms import Spectrum
from spec2vec import SpectrumDocument

from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.base_storer import BaseStorer
from spec2vec_mlops.helper_classes.embedding import Embedding
from spec2vec_mlops.helper_classes.feast_table import FeastTable

KEYS = config["cleaned_data"]["necessary_keys"]

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
            entity_description="spectrum_identifier"
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
                "words": ValueType.STRING_LIST,
                "losses": ValueType.STRING_LIST,
                "weights": ValueType.DOUBLE_LIST,
            },
        )
        self._document_table = self._feast_table.get_or_create_table(
            entity_description="document_identifier"
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
    def __init__(
        self, out_dir: str, feast_core_url: str, feature_table_name: str, run_id: str
    ):
        self._feast_table = FeastTable(
            out_dir,
            feast_core_url,
            feature_table_name,
            **{"run_id": ValueType.STRING, "embedding": ValueType.DOUBLE_LIST},
        )
        self.run_id = run_id
        self._embedding_table = self._feast_table.get_or_create_table(
            entity_description="embedding_identifier"
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
