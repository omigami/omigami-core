from prefect import Flow

from omigami.spectra_matching.ms2deepscore.tasks import (
    CreateSpectrumIDsChunks,
)
from omigami.spectra_matching.storage import RedisSpectrumDataGateway
from omigami.test.spectra_matching.conftest import TEST_TASK_CONFIG


def test_create_chunks(spectrum_ids):
    with Flow("test-flow") as test_flow:
        chunks = CreateSpectrumIDsChunks(
            chunk_size=10,
            spectrum_dgw=RedisSpectrumDataGateway(),
            **TEST_TASK_CONFIG,
        )(spectrum_ids)

    res = test_flow.run()

    assert res.is_successful()
    assert len(res.result[chunks].result) == 10
