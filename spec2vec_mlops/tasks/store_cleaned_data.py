import datetime
import logging
import feast
from typing import List
import pandas as pd
from feast import ValueType, FeatureTable, Feature, Entity, FileSource
from matchms import Spectrum
from prefect import task

from spec2vec_mlops import config

KEYS = config["cleaned_data"]["necessary_keys"].get(list)
FEAST_CORE_URL = config["feast"]["url"].get(str)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataStorer:
    def __init__(self, out_dir: str):
        self.feast_core_url = FEAST_CORE_URL
        self.feature_table_name = "spectrum_info"
        not_string_features2types = {
            "mz_list": ValueType.DOUBLE_LIST,
            "intensity_list": ValueType.DOUBLE_LIST,
            "precursor_mz": ValueType.FLOAT,
            "charge": ValueType.INT64,
            "parent_mass": ValueType.FLOAT,
        }
        string_features2types = {
            key.lower(): ValueType.STRING
            for key in KEYS
            if key.lower() not in not_string_features2types.keys()
        }
        self.features2types = {**not_string_features2types, **string_features2types}
        self.out_dir = out_dir

    def store_cleaned_data(self, data: List[Spectrum]):
        client = feast.Client(core_url=self.feast_core_url, telemetry=False)
        if not any(
            table.name != self.feature_table_name
            for table in client.list_feature_tables()
        ):
            spectrum_info = self._create_spectrum_info_table(client)
        else:
            spectrum_info = client.get_feature_table(self.feature_table_name)
        data_df = self._get_data_df(data)
        client.ingest(spectrum_info, data_df)

    def _create_spectrum_info_table(self, client) -> FeatureTable:
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
            file_format=feast.data_format.ParquetFormat(),
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
        client.apply(spectrum_id)
        client.apply(spectrum_info)
        return spectrum_info

    def _get_data_df(self, data: List[Spectrum]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "spectrum_id": spectrum.metadata["spectrum_id"],
                    "mz_list": spectrum.peaks.mz,
                    "intensity_list": spectrum.peaks.intensities,
                    **{
                        key: spectrum.metadata[key]
                        for key in self.features2types.keys()
                        if key in spectrum.metadata.keys()
                    },
                    "event_timestamp": datetime.datetime.strptime(
                        spectrum.metadata["create_time"], "%Y-%m-%d %H:%M:%S.%f"
                    ),
                    "created_timestamp": datetime.datetime.now(),
                }
                for spectrum in data
            ]
        )


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def store_cleaned_task(data: List[Spectrum], out_dir: str):
    ds = DataStorer(out_dir)
    ds.store_cleaned_data(data)
