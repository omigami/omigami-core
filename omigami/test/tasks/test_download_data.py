from unittest.mock import MagicMock

import pytest
from drfs import DRPath
from drfs.filesystems import get_fs
from prefect import Flow

from omigami.config import SOURCE_URI_PARTIAL_GNPS
from omigami.flows.utils import create_result
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.tasks import DownloadData
from omigami.data_gateway import SpectrumDataGateway
from omigami.tasks.download_data import DownloadParameters
from omigami.test.conftest import ASSETS_DIR, TEST_TASK_CONFIG


def test_download_data():
    input_dgw = MagicMock(spec=SpectrumDataGateway)
    input_dgw.download_gnps.return_value = "download"
    input_dgw.get_spectrum_ids.return_value = "spectrum_ids"
    download_params = DownloadParameters("input-uri", "dir", "file_name")
    with Flow("test-flow") as test_flow:
        download = DownloadData(
            input_dgw,
            **download_params.kwargs,
            **create_result(DRPath("")),
            **TEST_TASK_CONFIG,
        )()

    res = test_flow.run()

    assert res.is_successful()
    assert res.result[download].result == "spectrum_ids"
    input_dgw.download_gnps.assert_called_once_with(
        download_params.input_uri, download_params.download_path
    )
    input_dgw.get_spectrum_ids.assert_called_once_with(download_params.download_path)
    input_dgw.serialize_to_file.assert_called_once_with(
        download_params.checkpoint_path, "spectrum_ids"
    )


def test_download_existing_data():
    file_name = "SMALL_GNPS.json"
    input_dgw = FSInputDataGateway()
    input_dgw.download_gnps = lambda *args: None
    fs = get_fs(ASSETS_DIR)
    params = DownloadParameters(SOURCE_URI_PARTIAL_GNPS, ASSETS_DIR, file_name)

    with Flow("test-flow") as test_flow:
        download = DownloadData(
            input_dgw,
            **params.kwargs,
            **create_result(ASSETS_DIR / "spectrum_ids.pkl"),
            **TEST_TASK_CONFIG,
        )()

    res = test_flow.run()

    assert res.is_successful()
    assert fs.exists(DRPath(ASSETS_DIR) / file_name)
    assert res.result[download].is_cached()


@pytest.mark.skip(reason="This test uses internet connection.")
def test_download_existing_data_s3():
    file_name = "spec2vec-training-flow/downloaded_datasets/test_10k/gnps.json"
    dir_ = "s3://dr-prefect"
    bucket = "dr-prefect"
    checkpoint_name = (
        "spec2vec-training-flow/downloaded_datasets/test_10k/spectrum_ids.pkl"
    )
    input_dgw = FSInputDataGateway()
    fs = get_fs(dir_)
    download_params = DownloadParameters(
        SOURCE_URI_PARTIAL_GNPS, dir_, file_name, input_dgw, checkpoint_name
    )

    with Flow("test-flow") as test_flow:
        download = DownloadData(
            **download_params.kwargs,
            **create_result(download_params.download_path),
            **TEST_TASK_CONFIG,
        )()

    res = test_flow.run()

    assert res.is_successful()
    assert fs.exists(DRPath(bucket) / file_name)
    assert res.result[download].is_cached()
