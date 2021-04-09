import pandas as pd
from drfs.filesystems import get_fs


def read_parquet(uri) -> pd.DataFrame:
    fs = get_fs(uri)
    with fs.open(uri, "rb") as f:
        return pd.read_parquet(f)
