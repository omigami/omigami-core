import pytest

from spec2vec_mlops.helper_classes.feast_utils import FeastUtils


@pytest.fixture()
def parquet_uri(assets_dir):
    print(f"file://{str(assets_dir / 'feast_output.parquet')}")
    return f"file://{str(assets_dir / 'feast_output.parquet')}"


def test_read_parquet(parquet_uri):
    df = FeastUtils.read_parquet(parquet_uri)
    assert len(df) > 0
