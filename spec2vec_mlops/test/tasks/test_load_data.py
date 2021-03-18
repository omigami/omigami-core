import pytest

from spec2vec_mlops import config
from spec2vec_mlops.tasks.load_data import DataLoader

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
