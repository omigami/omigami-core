from pathlib import Path

import pytest
import requests_mock
from drfs import DRPath
from drfs.filesystems import get_fs

from spec2vec_mlops.config import default_configs
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway, KEYS


def test_load_gnps(local_gnps_small_json):
    for res in FSInputDataGateway().load_spectrum(local_gnps_small_json):
        assert isinstance(res, dict)
        for k in KEYS:
            assert k in res


SOURCE_URI_COMPLETE_GNPS = default_configs["gnps_json"]["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = default_configs["gnps_json"]["uri"]["partial"]


@pytest.mark.skip("Uses internet connection.")
@pytest.mark.slow
@pytest.mark.parametrize(
    "uri",
    [
        SOURCE_URI_PARTIAL_GNPS,
        SOURCE_URI_COMPLETE_GNPS,
    ],
)
def test_download_and_serialize_to_local(uri, tmpdir):
    _ = FSInputDataGateway().download_gnps(uri=uri, output_path=tmpdir / "test-ds")

    assert (tmpdir / "test-ds").exists()


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
