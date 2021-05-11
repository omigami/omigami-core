import pytest
from prefect.engine.results import S3Result, LocalResult

from spec2vec_mlops.flows.utils import create_result


def test_create_result():
    s3_path = "s3://my-bucket/file.txt"
    local_path = "/Users/batman/config.yaml"
    gcs_path = "gcs://nope"

    s3_res = create_result(s3_path)
    assert isinstance(s3_res, S3Result)
    assert s3_res.bucket == "my-bucket"

    local_res = create_result(local_path, validate_dir=False)
    assert isinstance(local_res, LocalResult)
    assert local_res.dir == "/Users/batman"

    with pytest.raises(KeyError):
        assert create_result(gcs_path)
