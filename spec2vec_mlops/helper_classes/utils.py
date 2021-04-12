import pandas as pd
from pyarrow.fs import FileSystem
from pyarrow.parquet import ParquetDataset


def read_parquet(uri) -> pd.DataFrame:
    fs, path = FileSystem.from_uri(uri)
    dataset = ParquetDataset(path_or_paths=path, filesystem=fs)
    table = dataset.read()
    return table.to_pandas()
