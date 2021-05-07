from unittest.mock import MagicMock

import pytest
from drfs import DRPath
from drfs.filesystems import get_fs
from prefect import Flow
from prefect.engine.serializers import JSONSerializer

from spec2vec_mlops import config
from spec2vec_mlops.flows.utils import create_result
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway
from spec2vec_mlops.tasks import DownloadData
from spec2vec_mlops.tasks.data_gateway import InputDataGateway
from spec2vec_mlops.tasks.download_data import DownloadParameters
from spec2vec_mlops.test.conftest import ASSETS_DIR, TEST_TASK_CONFIG

SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]


def test_download_data():
    input_dgw = MagicMock(spec=InputDataGateway)
    input_dgw.download_gnps.return_value = "download"
    download_params = DownloadParameters("input-uri", "dir", "file_name", input_dgw)
    with Flow("test-flow") as test_flow:
        download = DownloadData(
            **download_params.kwargs, result=create_result(DRPath(""))
        )()

    res = test_flow.run()

    assert res.is_successful()
    assert res.result[download].result == download_params.download_path
    input_dgw.download_gnps.assert_called_once_with("input-uri", "dir/file_name")


def test_download_existing_data():
    file_name = "SMALL_GNPS.json"
    input_dgw = FSInputDataGateway()
    fs = get_fs(ASSETS_DIR)

    with Flow("test-flow") as test_flow:
        download = DownloadData(
            input_dgw,
            SOURCE_URI_PARTIAL_GNPS,
            ASSETS_DIR,
            file_name,
            result=create_result(ASSETS_DIR, serializer=JSONSerializer()),
            **TEST_TASK_CONFIG,
        )()

    res = test_flow.run()

    assert res.is_successful()
    assert fs.exists(DRPath(ASSETS_DIR) / file_name)
    assert res.result[download].is_cached()


@pytest.mark.skip(reason="This test uses internet connection.")
def test_download_existing_data_s3():
    file_name = "test-dataset-download/gnps.json"
    dir = "s3://dr-prefect"
    bucket = "dr-prefect"
    input_dgw = FSInputDataGateway()
    fs = get_fs(dir)
    download_params = DownloadParameters(
        SOURCE_URI_PARTIAL_GNPS, dir, file_name, input_dgw
    )

    with Flow("test-flow") as test_flow:
        download = DownloadData(
            **download_params.kwargs,
            result=create_result(
                download_params.download_path, serializer=JSONSerializer()
            ),
        )()

    res = test_flow.run()

    assert res.is_successful()
    assert fs.exists(DRPath(bucket) / file_name)
    assert res.result[download].is_cached()
