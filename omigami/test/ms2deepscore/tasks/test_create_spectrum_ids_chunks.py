from omigami.ms2deepscore.tasks import ChunkingIDsParameters, CreateSpectrumIDsChunks
from omigami.test.conftest import TEST_TASK_CONFIG
from prefect import Flow


def test_create_chunks(spectrum_ids):
    chunking_parameters = ChunkingIDsParameters(10)
    with Flow("test-flow") as test_flow:
        chunks = CreateSpectrumIDsChunks(
            chunking_parameters=chunking_parameters,
            **TEST_TASK_CONFIG,
        )(spectrum_ids)

    res = test_flow.run()

    assert res.is_successful()
    assert len(res.result[chunks].result) == 10