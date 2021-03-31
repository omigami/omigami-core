import datetime
import os
import time

import pandas as pd
from feast import ValueType, Client, FeatureTable, Entity, Feature, FileSource
from feast.data_format import ParquetFormat
from feast.pyspark.abc import RetrievalJob
from matchms import Spectrum
from spec2vec import SpectrumDocument

from spec2vec_mlops import config

FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION = os.getenv(
    "FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION",
    config["feast"]["spark"]["output_location"].get(str),
)


class FeatureLoader:
    def __init__(self, feast_core_url: str):
        self.client = Client(core_url=feast_core_url, telemetry=False)
        self.feature_table_name = "spectrum_info"

    def load_documents(self) -> pd.DataFrame:
        entities_of_interest = pd.DataFrame(
            {
                # TODO: calling with a list of spectrum_id
                "spectrum_id": ["CCMSLIB00000001547"],
                "event_timestamp": [datetime.datetime.now()],
            }
        )
        job = self.client.get_historical_features(
            # TODO: specify columns needed
            ["spectrum_info:weights"],
            entity_source=entities_of_interest,
            output_location=FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION,
        )
        self.wait_for_job(job)
        df = pd.read_parquet(FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION)
        return df

    @staticmethod
    def wait_for_job(job: RetrievalJob):
        while job.get_status().name not in ("FAILED", "COMPLETED"):
            print(".", end="")
            time.sleep(0.5)
