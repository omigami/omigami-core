import os

import pytest
from omigami.ms2deepscore.config import BINNED_SPECTRUM_HASHES
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from pytest_redis import factories

from omigami.ms2deepscore.config import PROJECT_NAME

redis_db = factories.redisdb("redis_nooproc")


pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)


def test_write_binned_spectra(redis_db, binned_spectra):
    dgw = MS2DeepScoreRedisSpectrumDataGateway(PROJECT_NAME)
    dgw.write_binned_spectra(binned_spectra, "positive")

    assert redis_db.hlen(f"{BINNED_SPECTRUM_HASHES}_positive") == len(binned_spectra)


def test_read_binned_spectra(binned_spectra_stored, binned_spectra):
    dgw = MS2DeepScoreRedisSpectrumDataGateway(PROJECT_NAME)
    retrieved_binned_spectra = dgw.read_binned_spectra("positive")

    assert len(retrieved_binned_spectra) == len(binned_spectra)
