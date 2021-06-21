from pathlib import Path

import pytest
import requests_mock
from drfs import DRPath
from drfs.filesystems import get_fs

from omigami.spec2vec.config import (
    SOURCE_URI_PARTIAL_GNPS,
    SOURCE_URI_COMPLETE_GNPS,
)
from omigami.ms2deepscore.config import MS2DEEP_MODEL_URI
from omigami.gateways.input_data_gateway import FSInputDataGateway, KEYS
from omigami.test.conftest import ASSETS_DIR


def test_load_gnps(local_gnps_small_json):
    for res in FSInputDataGateway().load_spectrum(local_gnps_small_json):
        assert isinstance(res, dict)
        for k in KEYS:
            assert k in res


@pytest.mark.skip("Uses internet connection.")
@pytest.mark.slow
@pytest.mark.parametrize(
    "uri",
    [
        SOURCE_URI_PARTIAL_GNPS,
        SOURCE_URI_COMPLETE_GNPS,
    ],
)
def test_download_gnps_and_serialize_to_local(uri, tmpdir):
    _ = FSInputDataGateway().download_gnps(uri=uri, output_path=tmpdir / "test-ds")

    assert (tmpdir / "test-ds").exists()


@pytest.mark.skip("Uses internet connection.")
@pytest.mark.slow
def test_download_ms2deep_and_serialize_to_local(tmpdir):
    _ = FSInputDataGateway().download_ms2deep_model(
        uri=MS2DEEP_MODEL_URI, output_path=tmpdir / "test-ms2deepscore-model"
    )

    assert (tmpdir / "test-ms2deepscore-model").exists()


def test_download_and_serialize_to_remote(loaded_data, s3_mock):
    with requests_mock.Mocker() as m:
        m.get(SOURCE_URI_PARTIAL_GNPS, text="bac")
        _ = FSInputDataGateway().download_gnps(
            uri=SOURCE_URI_PARTIAL_GNPS,
            output_path="s3://test-bucket/test-ds",
        )
        assert DRPath("s3://test-bucket/test-ds").exists()


@pytest.mark.slow
@pytest.mark.skip("Uses internet connection.")
def test_resume_download(tmpdir, local_gnps_small_json):
    path = Path(local_gnps_small_json)
    existing_file_size = path.stat().st_size
    new_path = f"{tmpdir}/SMALL_GNPS_remaining.json"

    FSInputDataGateway(get_fs(path))._resume_download(
        existing_file_size, SOURCE_URI_COMPLETE_GNPS, new_path
    )

    updated_file_size = Path(new_path).stat().st_size

    assert updated_file_size


def test_get_spectrum_ids(local_gnps_small_json):
    ids = FSInputDataGateway().get_spectrum_ids(local_gnps_small_json)

    assert len(ids) == 100
    assert ids[0] == "CCMSLIB00000001547"


def test_load_spectrum_ids(local_gnps_small_json, spectrum_ids):
    spectrum_data = FSInputDataGateway().load_spectrum_ids(
        local_gnps_small_json, spectrum_ids[:10]
    )

    assert len(spectrum_data) == len(spectrum_ids[:10])
    assert set(spectrum_ids[:10]) == {d["SpectrumID"] for d in spectrum_data}


@pytest.mark.parametrize(
    "ion_mode, expected_chunk_files",
    [
        ("positive", 3),
        ("negative", 2),
    ],
)
def test_chunk_gnps_outputs(
    local_gnps_small_json, clean_chunk_files, ion_mode, expected_chunk_files
):
    dgw = FSInputDataGateway()
    fs = get_fs(ASSETS_DIR)

    dgw.chunk_gnps(local_gnps_small_json, chunk_size=150000, ion_mode=ion_mode)

    assert len(fs.ls(ASSETS_DIR / "chunks" / ion_mode)) == expected_chunk_files


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
    dgw = FSInputDataGateway()
    fs = get_fs(ASSETS_DIR)

    dgw.chunk_gnps(local_gnps_small_json, chunk_size=150000, ion_mode=ion_mode)

    paths = fs.ls(ASSETS_DIR / "chunks" / ion_mode)
    assert len(paths) == expected_chunk_files

    chunked_ids = []
    for p in paths:
        chunked_ids += dgw.get_spectrum_ids(str(p))

    assert set(chunked_ids) == set(spectrum_ids_by_mode[ion_mode])
