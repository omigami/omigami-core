import time
from typing import Union

import pandas as pd
from feast.pyspark.abc import RetrievalJob, SparkJob
from pyarrow.fs import FileSystem
from pyarrow.parquet import ParquetDataset

from spec2vec_mlops.helper_classes.exception import StorerLoadError


class FeastUtils:
    """Utility class for Feast."""

    @classmethod
    def wait_for_job(cls, job: Union[RetrievalJob, SparkJob]):
        while job.get_status().name not in ("FAILED", "COMPLETED"):
            print(".", end="")
            time.sleep(0.5)
        if job.get_status().name == "FAILED":
            raise StorerLoadError

    @classmethod
    def read_parquet(cls, uri: str) -> pd.DataFrame:
        fs, path = FileSystem.from_uri(uri)
        dataset = ParquetDataset(path_or_paths=path, filesystem=fs)
        table = dataset.read()
        return table.to_pandas()
