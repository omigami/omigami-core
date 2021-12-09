import pytest
from drfs.filesystems import get_fs
from prefect import Flow

from omigami.spectra_matching.storage import FSDataGateway
from omigami.spectra_matching.tasks import CreateChunks, ChunkingParameters
from omigami.test.spectra_matching.conftest import TEST_TASK_CONFIG, ASSETS_DIR


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
    data_gtw = FSDataGateway()
    fs = get_fs(local_gnps_small_json)
    output_directory = ASSETS_DIR / "raw" / ion_mode
    chunking_parameters = ChunkingParameters(
        local_gnps_small_json, str(output_directory), 150000, ion_mode
    )
    with Flow("test-flow") as test_flow:
        chunks = CreateChunks(
            data_gtw=data_gtw, chunking_parameters=chunking_parameters
        )(spectrum_ids)

    res = test_flow.run()
    res_2 = test_flow.run()

    assert res.is_successful()
    assert res_2.result[chunks].is_cached()
    assert fs.exists(output_directory / "chunk_paths.pickle")
    assert len(fs.ls(output_directory)) == expected_chunk_files + 1
    assert set(res.result[chunks].result) == set(res_2.result[chunks].result)


@pytest.mark.parametrize(
    "ion_mode, expected_chunk_files",
    [
        ("positive", 3),
        ("negative", 2),
    ],
)
def test_chunk_gnps_data_consistency(
    local_gnps_small_json,
    clean_chunk_files,
    ion_mode,
    expected_chunk_files,
    spectrum_ids_by_mode,
):
    data_gtw = FSDataGateway()
    output_directory = ASSETS_DIR / "raw" / ion_mode
    chunking_parameters = ChunkingParameters(
        local_gnps_small_json, str(output_directory), 150000, ion_mode
    )
    t = CreateChunks(
        data_gtw=data_gtw,
        chunking_parameters=chunking_parameters,
        **TEST_TASK_CONFIG,
    )

    t.chunk_gnps(local_gnps_small_json)

    paths = data_gtw.list_files(output_directory)
    assert len(paths) == expected_chunk_files

    chunked_ids = []
    for p in paths:
        chunked_ids += data_gtw.get_spectrum_ids(str(p))

    assert set(chunked_ids) == set(spectrum_ids_by_mode[ion_mode])
