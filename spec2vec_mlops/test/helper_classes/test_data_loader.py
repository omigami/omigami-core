import pytest
from pathlib import Path

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
def test_load_gnps_json_with_web_uri(uri):
    dl = DataLoader()

    res = dl.load_gnps_json(uri)

    assert isinstance(res, list)
    for r in res:
        assert isinstance(r, dict)


def test_load_gnps_json_with_local_uri(gnps_small_json):
    dl = DataLoader()

    for res in dl.load_gnps_json(gnps_small_json):
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
def test_save_web_uri(uri, tmpdir):
    dl = DataLoader(uri=uri)
    res = dl._save(out_dir=tmpdir)

    assert isinstance(res, str)
    assert res.split(".")[1] == "json"
    assert Path(res).exists()


@pytest.mark.longrun
@pytest.mark.parametrize(
    "uri",
    [
        "https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json",
        "https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json",
    ],
)
def test_load_web_uri(uri, tmpdir):
    dl = DataLoader(uri=uri)

    for res in dl.load(out_dir=tmpdir):
        assert isinstance(res, dict)
        for k in KEYS:
            assert k in res


def test_load_local_uri_save_to_s3(gnps_small_json):
    dl = DataLoader(uri=gnps_small_json)

    for res in dl.load():
        assert isinstance(res, dict)
        for k in KEYS:
            assert k in res
