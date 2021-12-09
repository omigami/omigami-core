import pytest
from prefect import Flow

from omigami.spectra_matching.storage import FSDataGateway
from omigami.spectra_matching.tasks import (
    ChunkingParameters,
    CreateChunks,
    CleanRawSpectraParameters,
    CleanRawSpectra,
)
from omigami.test.spectra_matching.conftest import ASSETS_DIR


@pytest.fixture
def create_chunks_task(clean_chunk_files, local_gnps_small_json):
    data_gtw = FSDataGateway()
    output_directory = ASSETS_DIR / "raw" / "positive"
    chunking_parameters = ChunkingParameters(
        local_gnps_small_json, str(output_directory), 150000, "positive"
    )
    t = CreateChunks(data_gtw=data_gtw, chunking_parameters=chunking_parameters)

    return t


@pytest.fixture
def clean_spectra_task():
    output_dir = ASSETS_DIR / "cleaned"
    fs_dgw = FSDataGateway()
    params = CleanRawSpectraParameters(output_dir)
    t = CleanRawSpectra(fs_dgw, params)

    return t


@pytest.fixture()
def cleaned_spectra_paths(create_chunks_task, clean_spectra_task):
    with Flow("test-flow") as flow:
        cc = create_chunks_task()
        cs = clean_spectra_task.map(cc)

    res = flow.run()

    return res.result[cs].result


@pytest.fixture
def cleaned_spectra(cleaned_spectra_paths):
    fs_dgw = FSDataGateway()
    return [fs_dgw.read_from_file(p) for p in cleaned_spectra_paths]
