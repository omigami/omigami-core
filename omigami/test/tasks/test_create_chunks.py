from prefect import Flow

from omigami.flows.utils import create_result
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.tasks import CreateChunks
from omigami.test.conftest import TEST_TASK_CONFIG, ASSETS_DIR


def test_create_chunks(local_gnps_small_json, spectrum_ids, clean_chunk_files):
    input_dgw = FSInputDataGateway()
    with Flow("test-flow") as test_flow:
        chunks = CreateChunks(
            local_gnps_small_json,
            input_dgw,
            25,
            **create_result(ASSETS_DIR / "chunk_paths.pickle"),
            **TEST_TASK_CONFIG,
        )(spectrum_ids)

    res = test_flow.run()
    res_2 = test_flow.run()

    assert res.is_successful()
    assert res_2.result[chunks].is_cached()
    assert input_dgw.fs.exists(ASSETS_DIR / "chunk_paths.pickle")
    assert len(input_dgw.fs.ls(ASSETS_DIR / "chunks")) == 4
    assert set(res.result[chunks].result) == set(res_2.result[chunks].result)
