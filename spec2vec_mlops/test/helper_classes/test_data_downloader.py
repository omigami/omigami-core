from pathlib import Path

import pytest

from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_downloader import DataDownloader

KEYS = config["gnps_json"]["necessary_keys"]

SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]


@pytest.fixture()
def data_downloader(tmpdir):
    return DataDownloader(tmpdir)


@pytest.mark.longrun
@pytest.mark.parametrize(
    "uri",
    [
        SOURCE_URI_PARTIAL_GNPS,
        SOURCE_URI_COMPLETE_GNPS,
    ],
)
def test_download_and_serialize_web_uri(uri, data_downloader):
    res = data_downloader.download_gnps_json(uri=uri)

    assert Path(res).exists()


def test_download_already_exists(data_downloader):
    file_path = data_downloader.download_gnps_json(uri=SOURCE_URI_PARTIAL_GNPS)
    same_file_path = data_downloader.download_gnps_json(uri=SOURCE_URI_PARTIAL_GNPS)

    assert file_path == same_file_path


@pytest.mark.longrun
def test_resume_download(data_downloader, tmpdir, local_gnps_small_json):
    path = Path(local_gnps_small_json)
    existing_file_size = path.stat().st_size
    new_path = f"{tmpdir}/SMALL_GNPS_remaining.json"

    data_downloader._resume_download(existing_file_size, SOURCE_URI_COMPLETE_GNPS, new_path)

    updated_file_size = Path(new_path).stat().st_size

    assert updated_file_size


def test_make_path(data_downloader):
    path = data_downloader._make_path()

    assert isinstance(path, str)
    assert path.split(".")[1] == "json"
