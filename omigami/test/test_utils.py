import pytest
from prefect.engine.results import S3Result, LocalResult

from omigami.config import DEFAULT_PREFECT_TASK_CONFIG
from omigami.utils import create_prefect_result_from_path, merge_prefect_task_configs


def test_create_prefect_result_from_path():
    s3_path = "s3://my-bucket/file/file2.txt"
    local_path = "/Users/batman/config.yaml"
    gcs_path = "gcs://nope"

    s3_res = create_prefect_result_from_path(s3_path)
    assert isinstance(s3_res["result"], S3Result)
    assert s3_res["result"].bucket == "my-bucket"
    assert s3_res["target"] == "file/file2.txt"

    local_res = create_prefect_result_from_path(local_path, validate_dir=False)
    assert isinstance(local_res["result"], LocalResult)
    assert local_res["result"].dir == "/Users/batman"
    assert local_res["target"] == "config.yaml"

    with pytest.raises(KeyError):
        assert create_prefect_result_from_path(gcs_path)


def test_merge_prefect_task_configs():
    kwargs = {"parameter_a": 1, "parameter_b": 2}

    params = merge_prefect_task_configs(kwargs)

    assert params == {**DEFAULT_PREFECT_TASK_CONFIG.copy(), **kwargs}
