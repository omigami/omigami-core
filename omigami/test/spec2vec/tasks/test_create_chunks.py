import pytest
from drfs.filesystems import get_fs
from prefect import Flow

from omigami.spec2vec.gateways.input_data_gateway import FSInputDataGateway
from omigami.spec2vec.tasks import CreateChunks, ChunkingParameters
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
    fs = get_fs(local_gnps_small_json)
    chunking_parameters = ChunkingParameters(local_gnps_small_json, 150000, ion_mode)
    with Flow("test-flow") as test_flow:
        chunks = CreateChunks(
            input_dgw=input_dgw,
            chunking_parameters=chunking_parameters,
            **TEST_TASK_CONFIG,
        )(spectrum_ids)

    res = test_flow.run()
    res_2 = test_flow.run()

    assert res.is_successful()
    assert res_2.result[chunks].is_cached()
    assert fs.exists(ASSETS_DIR / f"chunks/{ion_mode}/chunk_paths.pickle")
    assert len(fs.ls(ASSETS_DIR / "chunks" / ion_mode)) == expected_chunk_files + 1
    assert set(res.result[chunks].result) == set(res_2.result[chunks].result)
