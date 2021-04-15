from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from feast import ValueType
from matchms import Spectrum
from spec2vec import SpectrumDocument

from spec2vec_mlops import config
from spec2vec_mlops.entities.embedding import Embedding
from spec2vec_mlops.entities.feast_spectrum_document import FeastSpectrumDocument
from spec2vec_mlops.helper_classes.base_storer import BaseStorer
from spec2vec_mlops.helper_classes.exception import StorerLoadError
from spec2vec_mlops.helper_classes.feast_dgw import FeastDataGateway

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
KEYS = config["cleaned_data"]["necessary_keys"]
not_string_features2types = {
    "mz_list": ValueType.DOUBLE_LIST,
    "intensity_list": ValueType.DOUBLE_LIST,
    "losses": ValueType.DOUBLE_LIST,
    "precursor_mz": ValueType.DOUBLE,
    "charge": ValueType.INT64,
    "parent_mass": ValueType.DOUBLE,
}
string_features2types = {
    key.lower(): ValueType.STRING
    for key in KEYS
    if key.lower() not in not_string_features2types.keys()
}


class SpectrumIDStorer(BaseStorer):
    def __init__(self, feature_table_name: str):
        self._dgw = FeastDataGateway()
        self._table = self._dgw.get_or_create_table(
            entity_name="spectrum_ids_id",
            entity_description="List of spectrum IDs identifier",
            feature_table_name=feature_table_name,
            features2types={
                "all_spectrum_ids": ValueType.STRING_LIST,
            },
        )

    def store(self, data: List[str]):
        try:
            existing_ids = self.read_offline()
        except StorerLoadError:
            existing_ids = []
        new_ids = list(set(data).difference(existing_ids))
        all_ids = [*existing_ids, *new_ids]
        data_df = self._get_data_df(all_ids)
        self._dgw.store_offline(self._table, data_df)
        self._dgw.store_online(self._table)

    def read_offline(self) -> List[str]:
        feature_list = [f"{self._table.name}:all_spectrum_ids"]
        entity_dict = {
            "spectrum_ids_id": ["1"],
            "event_timestamp": [datetime.now()],
        }
        df = self._dgw.read_offline(feature_list, entity_dict)
        return df[f"{self._table.name}__all_spectrum_ids"].iloc[0]

    def read_online(self) -> List[str]:
        feature_list = [f"{self._table.name}:all_spectrum_ids"]
        entity_rows = [{"spectrum_ids_id": "1"}]
        df = self._dgw.read_online(feature_list, entity_rows)
        return df[f"{self._table.name}:all_spectrum_ids"][0]

    def _get_data_df(self, data: List[str]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "spectrum_ids_id": "1",
                    "all_spectrum_ids": data,
                    "event_timestamp": datetime.now(),
                    "created_timestamp": datetime.now(),
                }
            ]
        )


class SpectrumStorer(BaseStorer):
    def __init__(self, feature_table_name: str):
        self._dgw = FeastDataGateway()
        self._table = self._dgw.get_or_create_table(
            entity_name="spectrum_id",
            entity_description="spectrum_identifier",
            feature_table_name=feature_table_name,
            features2types={
                **string_features2types,
                **not_string_features2types,
                **{"metadata_timestamp": ValueType.STRING},
            },
        )

    def store(self, data: List[Spectrum]) -> List[str]:
        data_df = self._get_data_df(data)
        self._dgw.store_offline(self._table, data_df)
        self._dgw.store_online(self._table)
        return data_df["spectrum_id"].tolist()

    def read_offline(self, ids: List[str]) -> List[Spectrum]:
        feature_list = [
            f"{self._table.name}:mz_list",
            f"{self._table.name}:intensity_list",
            f"{self._table.name}:losses",
            f"{self._table.name}:metadata_timestamp",
        ]
        entity_dict = {
            "spectrum_id": ids,
            "event_timestamp": [datetime.now()] * len(ids),
        }
        df = self._dgw.read_offline(feature_list, entity_dict)
        df = df.set_index("spectrum_id")
        return self._get_output(df, sep="__")

    def read_online(self, ids: List[str]) -> List[Spectrum]:
        feature_list = [
            f"{self._table.name}:mz_list",
            f"{self._table.name}:intensity_list",
            f"{self._table.name}:losses",
            f"{self._table.name}:metadata_timestamp",
        ]
        entity_rows = [{"spectrum_id": id} for id in ids]
        df = self._dgw.read_online(feature_list, entity_rows)
        df = df.set_index("spectrum_id")
        return self._get_output(df, sep=":")

    def _get_data_df(self, data: List[Spectrum]) -> pd.DataFrame:
        features2types = {
            **string_features2types,
            **not_string_features2types,
            **{"metadata_timestamp": ValueType.STRING},
        }
        return pd.DataFrame.from_records(
            [
                {
                    "spectrum_id": spectrum.metadata["spectrum_id"],
                    "mz_list": spectrum.peaks.mz,
                    "intensity_list": spectrum.peaks.intensities,
                    "losses": spectrum.losses
                    or [np.inf],  # need this to maintain the correct type in Spark
                    **{
                        key: spectrum.metadata[key]
                        for key in features2types.keys()
                        if key in spectrum.metadata.keys()
                    },
                    "metadata_timestamp": spectrum.metadata.get("create_time"),
                    "event_timestamp": self._convert_create_time(
                        spectrum.metadata.get("create_time")
                    ),
                    "created_timestamp": datetime.now(),
                }
                for spectrum in data
            ]
        )

    @staticmethod
    def _get_output(df: pd.DataFrame, sep: str = ":") -> List[Spectrum]:
        spectra = []
        for spectrum_id, record in df.iterrows():
            spectrum = Spectrum(
                mz=record[f"spectrum_info{sep}mz_list"],
                intensities=record[f"spectrum_info{sep}intensity_list"],
                metadata={
                    "spectrum_id": spectrum_id,
                    "create_time": record[f"spectrum_info{sep}metadata_timestamp"],
                },
            )
            if record[f"spectrum_info{sep}losses"] == [np.inf]:
                spectrum.losses = None
            else:
                spectrum.losses = record[f"spectrum_info{sep}losses"]
            spectra.append(spectrum)
        return spectra

    @staticmethod
    def _convert_create_time(create_time: str) -> datetime:
        if create_time:
            return datetime.strptime(create_time, TIME_FORMAT)
        else:
            return datetime.now()


class DocumentStorer(BaseStorer):
    def __init__(self, feature_table_name: str):
        self._dgw = FeastDataGateway()
        self._table = self._dgw.get_or_create_table(
            entity_name="spectrum_id",
            entity_description="document_identifier",
            feature_table_name=feature_table_name,
            features2types={
                "words": ValueType.STRING_LIST,
                "losses": ValueType.STRING_LIST,
                "weights": ValueType.DOUBLE_LIST,
                "n_decimals": ValueType.INT64,
            },
        )

    def store(self, data: List[SpectrumDocument]):
        data_df = self._get_data_df(data)
        self._dgw.store_offline(self._table, data_df)
        self._dgw.store_online(self._table)

    def read_offline(self, ids: List[str]) -> List[FeastSpectrumDocument]:
        feature_list = [
            f"{self._table.name}:words",
            f"{self._table.name}:losses",
            f"{self._table.name}:weights",
            f"{self._table.name}:n_decimals",
        ]
        entity_dict = {
            "spectrum_id": ids,
            "event_timestamp": [datetime.now()] * len(ids),
        }
        df = self._dgw.read_offline(feature_list, entity_dict)
        df = df.set_index("spectrum_id")
        return self._get_output(df, sep="__")

    def read_online(self, ids: List[str]) -> List[FeastSpectrumDocument]:
        feature_list = [
            f"{self._table.name}:words",
            f"{self._table.name}:losses",
            f"{self._table.name}:weights",
            f"{self._table.name}:n_decimals",
        ]
        entity_rows = [{"spectrum_id": id} for id in ids]
        df = self._dgw.read_online(feature_list, entity_rows)
        df = df.set_index("spectrum_id")
        return self._get_output(df, sep=":")

    def _get_data_df(self, data: List[SpectrumDocument]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "spectrum_id": document.metadata["spectrum_id"],
                    "words": document.words,
                    "losses": document.losses
                    or [""]
                    * len(
                        document.words
                    ),  # needed to maintain the correct type in Spark
                    "weights": document.weights,
                    "n_decimals": document.n_decimals,
                    "event_timestamp": self._convert_create_time(
                        document.metadata.get("create_time")
                    ),
                    "created_timestamp": datetime.now(),
                }
                for document in data
            ]
        )

    @staticmethod
    def _get_output(df: pd.DataFrame, sep: str = ":") -> List[FeastSpectrumDocument]:
        documents = []
        for spectrum_id, record in df.iterrows():
            doc = FeastSpectrumDocument(
                {
                    "words": record[f"document_info{sep}words"],
                    "losses": record[f"document_info{sep}losses"],
                    "weights": record[f"document_info{sep}weights"],
                    "n_decimals": record[f"document_info{sep}n_decimals"],
                    "metadata": {"spectrum_id": spectrum_id},
                }
            )
            documents.append(doc)
        return documents

    @staticmethod
    def _convert_create_time(create_time: str) -> datetime:
        if create_time:
            return datetime.strptime(create_time, TIME_FORMAT)
        else:
            return datetime.now()


class EmbeddingStorer(BaseStorer):
    def __init__(self, feature_table_name: str, run_id: str):
        self._dgw = FeastDataGateway()
        self.run_id = run_id
        self._table = self._dgw.get_or_create_table(
            entity_name="spectrum_id",
            entity_description="embedding_identifier",
            feature_table_name=feature_table_name,
            features2types={
                "run_id": ValueType.STRING,
                "embedding": ValueType.DOUBLE_LIST,
                "n_decimals": ValueType.INT64,
            },
        )

    def store(self, data: List[Embedding]):
        data_df = self._get_data_df(data)
        self._dgw.store_offline(self._table, data_df)
        self._dgw.store_online(self._table)

    def read_offline(self, ids: List[str]) -> List[Embedding]:
        feature_list = [
            f"{self._table.name}:embedding",
            f"{self._table.name}:n_decimals",
        ]
        entity_dict = {
            "spectrum_id": ids,
            "run_id": [self.run_id] * len(ids),
            "event_timestamp": [datetime.now()] * len(ids),
        }
        df = self._dgw.read_offline(feature_list, entity_dict)
        df = df.set_index("spectrum_id")
        return self._get_output(df, sep="__")

    def read_online(self, ids: List[str]) -> List[Embedding]:
        feature_list = [
            f"{self._table.name}:embedding",
            f"{self._table.name}:n_decimals",
        ]
        entity_rows = [{"spectrum_id": id} for id in ids]
        df = self._dgw.read_online(feature_list, entity_rows)
        df = df.set_index("spectrum_id")
        return self._get_output(df, sep=":")

    def _get_data_df(self, embeddings: List[Embedding]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "spectrum_id": embedding.spectrum_id,
                    "embedding": embedding.vector,
                    "run_id": self.run_id,
                    "n_decimals": embedding.n_decimals,
                    "event_timestamp": datetime.now(),
                    "created_timestamp": datetime.now(),
                }
                for embedding in embeddings
            ]
        )

    @staticmethod
    def _get_output(df: pd.DataFrame, sep: str = ":") -> List[Embedding]:
        embeddings = []
        for spectrum_id, record in df.iterrows():
            embeddings.append(
                Embedding(
                    vector=record[f"embedding_info{sep}embedding"],
                    spectrum_id=spectrum_id,
                    n_decimals=record[f"embedding_info{sep}n_decimals"],
                )
            )
        return embeddings
