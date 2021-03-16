import datetime
import logging
import feast
from typing import List, Dict

from feast import ValueType
from matchms import Spectrum
from prefect import task

from spec2vec_mlops import config

KEYS = config["gnps_json"]["necessary_keys"].get(list)
FEAST_CORE_URL = config["feast"]["url"].get(str)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataStorer:
    def __init__(self, out_dir: str):
        self.features2types = {
            "mz_list": feast.value_type.ValueType.DOUBLE_LIST,
            "intensity_list": feast.value_type.ValueType.DOUBLE_LIST,
            "charge": feast.value_type.ValueType.INT64,
            "ionmode": feast.value_type.ValueType.STRING,
            "compound_name": feast.value_type.ValueType.STRING,
            "adduct": feast.value_type.ValueType.STRING,
            "formula_smiles": feast.value_type.ValueType.STRING,
            "precursor_mz": feast.value_type.ValueType.FLOAT,
            "inchikey": feast.value_type.ValueType.STRING,
            "smiles": feast.value_type.ValueType.STRING,
            "create_time": feast.value_type.ValueType.STRING,
        }
        self.out_dir = out_dir

    def store_cleaned_data(self, data: List[Spectrum]):
        client = feast.Client(core_url=FEAST_CORE_URL, telemetry=False)
        if not any(table.name != "spectrum_info" for table in client.list_feature_tables()):
            spectrum_info = self._create_spectrum_info_table(client)

    def _create_spectrum_info_table(self, client) -> feast.feature_table.FeatureTable:
        spectrum_id = feast.entity.Entity(
            name="spectrum_id",
            description="Spectrum identifier",
            value_type=ValueType.INT64,
        )
        features = [
            feast.feature.Feature(feature, dtype=feature_type)
            for feature, feature_type in self.features2types.items()
        ]
        batch_source = feast.data_source.FileSource(
            file_format=feast.data_format.ParquetFormat(),
            file_url=str(self.out_dir),
            event_timestamp_column="event_timestamp",
            created_timestamp_column="created_timestamp",
        )
        spectrum_info = feast.feature_table.FeatureTable(
            name="spectrum_info",
            entities=["spectrum_id"],
            features=features,
            batch_source=batch_source,
        )
        client.apply(spectrum_id)
        client.apply(spectrum_info)
        return spectrum_info


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def load_data_task(data: List[Spectrum], out_dir: str):
    ds = DataStorer(out_dir)
    ds.store_cleaned_data(data)
