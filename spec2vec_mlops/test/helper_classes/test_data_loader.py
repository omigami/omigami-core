from pathlib import Path

import pytest

from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_loader import DataLoader

KEYS = config["gnps_json"]["necessary_keys"].get(list)


@pytest.fixture()
def data_loader(tmpdir):
    return DataLoader(tmpdir)


@pytest.mark.longrun
@pytest.mark.parametrize(
    "uri",
    [
        "https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json",
        "https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json",
    ],
)
def test_load_gnps_json_with_web_uri(uri, data_loader):
    res = data_loader.load_gnps_json(uri)

    assert isinstance(res, list)
    for r in res:
        assert isinstance(r, dict)


def test_parse_json(local_gnps_small_json, data_loader):
    for res in data_loader.parse_json(local_gnps_small_json):
        assert isinstance(res, dict)
        for k in KEYS:
            assert k in res


@pytest.mark.longrun
@pytest.mark.parametrize(
    "uri",
    [
        "https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json",
        "https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json",
    ],
)
def test_download_and_serialize_web_uri(uri, data_loader):
    res = data_loader._download_and_serialize(uri=uri)

    assert Path(res).exists()


@pytest.mark.longrun
@pytest.mark.parametrize(
    "uri",
    [
        "https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json",
    ],
)
def test_resume_download(data_loader, tmpdir, uri, local_gnps_small_json):
    path = Path(local_gnps_small_json)
    existing_file_size = path.stat().st_size
    new_path = f"{tmpdir}/SMALL_GNPS_remaining.json"

    data_loader._resume_download(existing_file_size, uri, new_path)

    updated_file_size = Path(new_path).stat().st_size

    assert updated_file_size


def test_make_path(data_loader):
    res = data_loader._make_path()

    assert isinstance(res, str)
    assert res.split(".")[1] == "json"
