import datetime
import os
import time
from unittest.mock import MagicMock

import pytest
from drfs import DRPath
from drfs.filesystems import get_fs
from prefect import Flow

from omigami.config import SOURCE_URI_PARTIAL_GNPS
from omigami.spectra_matching.storage import DataGateway, FSDataGateway
from omigami.spectra_matching.tasks import DownloadParameters, DownloadData
from omigami.test.spectra_matching.conftest import ASSETS_DIR
from omigami.utils import create_prefect_result_from_path


def test_refresh_data(mock_default_config, tmpdir):
    """
    Test if the file is present, younger then 30 days and older then 30 days.
    """
    # Setting up test
    file_name = "file_name"

    data_gtw = MagicMock(spec=DataGateway)
    download_params = DownloadParameters("input-uri", tmpdir, file_name, "checkpoint")

    download = DownloadData(
        data_gtw,
        download_params,
    )

    file_dir = f"{tmpdir}/test"

    # Running function and collecting results
    # Check if file is present
    is_no_file = download.refresh_download(file_dir)

    open(file_dir, "a").close()

    # Check if file is younger then 30 days
    is_young_file = download.refresh_download(file_dir)

    # Changing file creation date to 2017
    date = datetime.datetime(year=2017, month=11, day=5, hour=19, minute=50, second=20)
    modTime = time.mktime(date.timetuple())
    os.utime(file_dir, (modTime, modTime))

    # check if file is older then 30 days
    is_old_file = download.refresh_download(file_dir)

    # Evaluating results
    assert is_no_file
    assert not is_young_file
    assert is_old_file


def test_download_data(mock_default_config, tmpdir):
    data_gtw = MagicMock(spec=FSDataGateway)
    data_gtw.download_gnps.return_value = "download"
    data_gtw.get_spectrum_ids.return_value = "spectrum_ids"
    download_params = DownloadParameters("input-uri", tmpdir, "file_name", "checkpoint")

    with Flow("test-flow") as test_flow:
        download = DownloadData(
            data_gtw,
            download_params,
        )()
        download.checkpointing = False
    res = test_flow.run()
    return_res = res.result[download].result
    assert res.is_successful()

    assert len(return_res) == 12
    data_gtw.download_gnps.assert_called_once_with(
        download_params.input_uri, download_params.download_path
    )
    data_gtw.get_spectrum_ids.assert_called_once_with(download_params.download_path)
    data_gtw.serialize_to_file.assert_called_once_with(
        download_params.checkpoint_path, "spectrum_ids"
    )


def test_download_existing_data(mock_default_config):
    file_name = "SMALL_GNPS.json"
    data_gtw = FSDataGateway()
    data_gtw.download_gnps = lambda *args: None
    fs = get_fs(ASSETS_DIR)
    params = DownloadParameters(
        SOURCE_URI_PARTIAL_GNPS,
        ASSETS_DIR.parent,
        ASSETS_DIR.name,
        dataset_file=file_name,
    )

    with Flow("test-flow") as test_flow:
        download = DownloadData(
            data_gtw,
            params,
        )()

    res = test_flow.run()

    assert res.is_successful()
    assert fs.exists(DRPath(ASSETS_DIR) / file_name)
    assert res.result[download].is_cached()


@pytest.mark.skip(reason="This test uses internet connection.")
def test_download_existing_data_s3(mock_default_config):
    file_name = "spec2vec-training-flow/downloaded_datasets/test_10k/gnps.json"
    dir_ = "s3://dr-prefect"
    bucket = "dr-prefect"
    checkpoint_name = (
        "spec2vec-training-flow/downloaded_datasets/test_10k/spectrum_ids.pkl"
    )
    data_gtw = FSDataGateway()
    fs = get_fs(dir_)
    download_params = DownloadParameters(
        SOURCE_URI_PARTIAL_GNPS, dir_, file_name, checkpoint_name
    )

    with Flow("test-flow") as test_flow:
        download = DownloadData(
            data_gtw,
            download_params,
            **create_prefect_result_from_path(download_params.download_path),
        )()

    res = test_flow.run()

    assert res.is_successful()
    assert fs.exists(DRPath(bucket) / file_name)
    assert res.result[download].is_cached()
