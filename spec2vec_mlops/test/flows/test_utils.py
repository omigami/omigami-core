import pytest
from prefect.engine.results import S3Result, LocalResult

from spec2vec_mlops.flows.utils import create_result


def test_create_result():
    s3_path = "s3://my-bucket/file/file2.txt"
    local_path = "/Users/batman/config.yaml"
    gcs_path = "gcs://nope"

    s3_res = create_result(s3_path)
    assert isinstance(s3_res["result"], S3Result)
    assert s3_res["result"].bucket == "my-bucket"
    assert s3_res["target"] == "file/file2.txt"

    local_res = create_result(local_path, validate_dir=False)
    assert isinstance(local_res["result"], LocalResult)
    assert local_res["result"].dir == "/Users/batman"
    assert local_res["target"] == "config.yaml"

    with pytest.raises(KeyError):
        assert create_result(gcs_path)
