import os
import pytest
from pytest_redis import factories
from omigami.ms2deepscore.config import BINNED_SPECTRUM_HASHES
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)

redis_db = factories.redisdb("redis_nooproc")


pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)


def test_write_binned_spectra(redis_db, binned_spectra):
    dgw = MS2DeepScoreRedisSpectrumDataGateway()
    dgw.write_binned_spectra(binned_spectra)

    assert redis_db.hlen(BINNED_SPECTRUM_HASHES) == len(binned_spectra)


def test_read_binned_spectra(binned_spectra_stored, binned_spectra):
    dgw = MS2DeepScoreRedisSpectrumDataGateway()
    retrieved_binned_spectra = dgw.read_binned_spectra()

    assert len(retrieved_binned_spectra) == len(binned_spectra)