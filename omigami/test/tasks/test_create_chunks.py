import pytest
from prefect import Flow

from omigami.flows.utils import create_result
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.tasks import CreateChunks
from omigami.test.conftest import TEST_TASK_CONFIG, ASSETS_DIR


@pytest.mark.parametrize(
    "ion_mode, expected_chunk_files",
    [
        ("positive", 3),
        ("negative", 2),
    ],
)
def test_create_chunks(
    local_gnps_small_json,
    spectrum_ids,
    clean_chunk_files,
    ion_mode,
    expected_chunk_files,
):
    input_dgw = FSInputDataGateway()
    with Flow("test-flow") as test_flow:
        chunks = CreateChunks(
            file_path=local_gnps_small_json,
            input_dgw=input_dgw,
            chunk_size=150000,
            ion_mode=ion_mode,
            **create_result(ASSETS_DIR / f"{ion_mode}/chunk_paths.pickle"),
            **TEST_TASK_CONFIG,
        )(spectrum_ids)

    res = test_flow.run()
    res_2 = test_flow.run()

    assert res.is_successful()
    assert res_2.result[chunks].is_cached()
    assert input_dgw.fs.exists(ASSETS_DIR / f"{ion_mode}/chunk_paths.pickle")
    assert (
        len(input_dgw.fs.ls(ASSETS_DIR / "chunks" / ion_mode)) == expected_chunk_files
    )
    assert set(res.result[chunks].result) == set(res_2.result[chunks].result)
