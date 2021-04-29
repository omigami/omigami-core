import os
import pickle

import pytest
from pytest_redis import factories

from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_loader import DataLoader

KEYS = config["gnps_json"]["necessary_keys"]
SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET = config["redis"]["spectrum_id_sorted_set"]
SPECTRUM_HASHES = config["redis"]["spectrum_hashes"]

redis_db = factories.redisdb("redis_nooproc")


@pytest.fixture()
def data_loader(local_gnps_small_json):
    return DataLoader(local_gnps_small_json)


def test_load_gnps_json(local_gnps_small_json, data_loader):
    for res in data_loader.load_gnps_json(ionmode="positive", skip_if_exists=True):
        assert isinstance(res, dict)
        for k in KEYS:
            assert k in res


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_load_gnps_json_skip_existing(
    local_gnps_small_json, data_loader, redis_db, cleaned_data
):
    pipe = redis_db.pipeline()
    for spectrum in cleaned_data:
        pipe.zadd(
            SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
            {spectrum.metadata["spectrum_id"]: spectrum.metadata["precursor_mz"]},
        )
        pipe.hset(
            SPECTRUM_HASHES, spectrum.metadata["spectrum_id"], pickle.dumps(spectrum)
        )
    pipe.execute()

    results = data_loader.load_gnps_json(ionmode="positive", skip_if_exists=True)
    assert len(results) == 0
