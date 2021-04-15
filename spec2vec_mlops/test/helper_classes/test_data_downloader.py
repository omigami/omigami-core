from pathlib import Path

import pytest

from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_downloader import DataDownloader

KEYS = config["gnps_json"]["necessary_keys"].get(list)

small_dataset_uri = "https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json"
complete_dataset_uri = "https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json"


@pytest.fixture()
def data_downloader(tmpdir):
    return DataDownloader(tmpdir)


@pytest.mark.longrun
@pytest.mark.parametrize(
    "uri",
    [
        small_dataset_uri,
        complete_dataset_uri,
    ],
)
def test_download_and_serialize_web_uri(uri, data_downloader):
    res = data_downloader.download_gnps_json(uri=uri)

    assert Path(res).exists()


def test_download_already_exists(data_downloader):
    file_path = data_downloader.download_gnps_json(uri=small_dataset_uri)
    same_file_path = data_downloader.download_gnps_json(uri=small_dataset_uri)

    assert file_path == same_file_path


@pytest.mark.longrun
def test_resume_download(data_downloader, tmpdir, local_gnps_small_json):
    path = Path(local_gnps_small_json)
    existing_file_size = path.stat().st_size
    new_path = f"{tmpdir}/SMALL_GNPS_remaining.json"

    data_downloader._resume_download(existing_file_size, complete_dataset_uri, new_path)

    updated_file_size = Path(new_path).stat().st_size

    assert updated_file_size


def test_make_path(data_downloader):
    path = data_downloader._make_path()

    assert isinstance(path, str)
    assert path.split(".")[1] == "json"
