import os

import pytest
from prefect import Flow

from omigami.gateways import RedisSpectrumDataGateway
from omigami.ms2deepscore.tasks import ChunkingParameters, CreateSpectrumIDsChunks
from omigami.test.conftest import TEST_TASK_CONFIG


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_create_chunks(spectra_stored):
    spectrum_dgw = RedisSpectrumDataGateway()
    chunking_parameters = ChunkingParameters(10)
    with Flow("test-flow") as test_flow:
        chunks = CreateSpectrumIDsChunks(
            spectrum_dgw=spectrum_dgw,
            chunking_parameters=chunking_parameters,
            **TEST_TASK_CONFIG,
        )()

    res = test_flow.run()

    assert res.is_successful()
    assert len(res.result[chunks].result) == 10
