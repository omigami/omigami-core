import pytest

from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_loader import DataLoader

KEYS = config["gnps_json"]["necessary_keys"].get(list)


@pytest.fixture()
def data_loader(local_gnps_small_json):
    return DataLoader(local_gnps_small_json)


def test_load_gnps_json(local_gnps_small_json, data_loader):
    for res in data_loader.load_gnps_json():
        assert isinstance(res, dict)
        for k in KEYS:
            assert k in res
