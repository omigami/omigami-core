from pathlib import Path

import pytest
from drfs.filesystems.local import LocalFileSystem

from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_loader import DataLoader

KEYS = config["gnps_json"]["necessary_keys"].get(list)


@pytest.mark.longrun
@pytest.mark.parametrize(
    "uri",
    [
        "https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json",
        "https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json",
    ],
)
def test_load_gnps_json_with_web_uri(uri, tmpdir):
    dl = DataLoader()

    res = dl.load_gnps_json(uri, tmpdir)

    assert isinstance(res, list)
    for r in res:
        assert isinstance(r, dict)


def test_parse_json(local_gnps_small_json, tmpdir):
    dl = DataLoader()
    fs = LocalFileSystem()

    for res in dl.parse_json(fs, local_gnps_small_json):
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
def test_download_and_serialize_web_uri(uri, tmpdir):
    dl = DataLoader()
    fs = LocalFileSystem()
    res = dl._download_and_serialize(uri=uri, fs=fs, out_dir=tmpdir)

    assert Path(res).exists()

@pytest.mark.longrun
@pytest.mark.parametrize(
    "uri",
    [
        "https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json",
    ],
)
def test_resume_download(tmpdir, uri, local_gnps_small_json):
    path = Path(local_gnps_small_json)
    existing_file_size = path.stat().st_size
    fs = LocalFileSystem()
    new_path = f"{tmpdir}/SMALL_GNPS_remaining.json"

    dl = DataLoader()
    dl._resume_download(existing_file_size, uri, fs, new_path)

    updated_file_size = Path(new_path).stat().st_size

    assert updated_file_size


def test_make_path(tmpdir):
    dl = DataLoader()

    res = dl._make_path(tmpdir)

    assert isinstance(res, str)
    assert res.split(".")[1] == "json"
