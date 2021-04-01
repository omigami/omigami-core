import datetime
import os
import time
from typing import List

import pandas as pd
from feast import Client
from feast.pyspark.abc import RetrievalJob
from matchms import Spectrum
from spec2vec import SpectrumDocument

from spec2vec_mlops import config

FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION = os.getenv(
    "FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION",
    config["feast"]["spark"]["output_location"].get(str),
)


class FeatureLoaderError(Exception):
    pass


class FeatureLoader:
    def __init__(self, feast_core_url: str):
        self.client = Client(core_url=feast_core_url, telemetry=False)
        self.feature_entity_name = "spectrum_id"
        self.meta_entity_name = "spectrum_meta_id"
        self.feature_table_name = "spectrum_info"
        self.meta_table_name = "spectrum_meta"

    def load_all_spectrum_ids(self) -> List[str]:
        entities_of_interest = pd.DataFrame(
            {
                self.meta_entity_name: [1],
                "event_timestamp": [datetime.datetime.now()],
            }
        )
        job = self.client.get_historical_features(
            [f"{self.meta_table_name}:all_spectrum_ids"],
            entity_source=entities_of_interest,
            output_location=f"file://{FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION}",
        )
        self.wait_for_job(job)
        if job.get_status().name == "FAILED":
            raise FeatureLoaderError
        df = pd.read_parquet(FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION)
        return df[f"{self.meta_table_name}__all_spectrum_ids"].iloc[0]

    def load_clean_data(self, spectrum_ids: List[str]) -> List[Spectrum]:
        entities_of_interest = pd.DataFrame(
            {
                self.feature_entity_name: spectrum_ids,
                "event_timestamp": [datetime.datetime.now()] * len(spectrum_ids),
            }
        )

        job = self.client.get_historical_features(
            [
                "spectrum_info:mz_list",
                "spectrum_info:intensity_list",
            ],
            entity_source=entities_of_interest,
            output_location=f"file://{FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION}",
        )
        self.wait_for_job(job)
        if job.get_status().name == "FAILED":
            raise FeatureLoaderError
        df = pd.read_parquet(FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION)
        df = df.set_index("spectrum_id")
        spectra = []
        for spectrum_id, record in df.iterrows():
            spectrum = Spectrum(
                mz=record["spectrum_info__mz_list"],
                intensities=record["spectrum_info__intensity_list"],
                metadata={"spectrum_id": spectrum_id},
            )
            spectra.append(spectrum)
        return spectra

    def load_documents(self, spectrum_ids: List[str]) -> List[SpectrumDocument]:
        entities_of_interest = pd.DataFrame(
            {
                self.feature_entity_name: spectrum_ids,
                "event_timestamp": [datetime.datetime.now()] * len(spectrum_ids),
            }
        )
        job = self.client.get_historical_features(
            [
                "spectrum_info:words",
            ],
            entity_source=entities_of_interest,
            output_location=f"file://{FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION}",
        )
        self.wait_for_job(job)
        if job.get_status().name == "FAILED":
            raise FeatureLoaderError
        df = pd.read_parquet(FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION)
        documents = []
        for spectrum_id, record in df.iterrows():
            # TODO: need to confirm if this is enough for Gensim
            documents.append(record["spectrum_info__words"])
        return documents

    @staticmethod
    def wait_for_job(job: RetrievalJob):
        while job.get_status().name not in ("FAILED", "COMPLETED"):
            print(".", end="")
            time.sleep(0.5)
