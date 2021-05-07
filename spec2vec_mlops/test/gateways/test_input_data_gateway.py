from pathlib import Path

import pytest
import requests_mock
from drfs import DRPath

from spec2vec_mlops import config
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway

KEYS = config["gnps_json"]["necessary_keys"]


def test_load_gnps(local_gnps_small_json):
    for res in FSInputDataGateway().load_spectrum(local_gnps_small_json):
        assert isinstance(res, dict)
        for k in KEYS:
            assert k in res


SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]


@pytest.mark.longrun
@pytest.mark.parametrize(
    "uri",
    [
        SOURCE_URI_PARTIAL_GNPS,
        SOURCE_URI_COMPLETE_GNPS,
    ],
)
def test_download_and_serialize_to_local(uri, tmpdir):
    res = FSInputDataGateway().download_gnps(uri=uri, output_path=tmpdir / "test-ds")

    assert Path(res).exists()


def test_download_and_serialize_to_remote(loaded_data, s3_mock):
    with requests_mock.Mocker() as m:
        m.get(SOURCE_URI_PARTIAL_GNPS, text="bac")
        res = FSInputDataGateway().download_gnps(
            uri=SOURCE_URI_PARTIAL_GNPS,
            output_path="s3://test-bucket/test-ds",
        )
        assert DRPath("s3://test-bucket/test-ds").exists()


@pytest.mark.longrun
def test_resume_download(tmpdir, local_gnps_small_json):
    path = Path(local_gnps_small_json)
    existing_file_size = path.stat().st_size
    new_path = f"{tmpdir}/SMALL_GNPS_remaining.json"

    FSInputDataGateway()._resume_download(
        existing_file_size, SOURCE_URI_COMPLETE_GNPS, new_path
    )

    updated_file_size = Path(new_path).stat().st_size

    assert updated_file_size


def test_get_spectrum_ids(local_gnps_small_json):
    ids = FSInputDataGateway().get_spectrum_ids(local_gnps_small_json)

    assert len(ids) == 100
    assert ids[0] == "CCMSLIB00000001547"


def test_load_spectrum_ids(local_gnps_small_json):
    spectrum_ids = [
        "CCMSLIB00000001547",
        "CCMSLIB00000001557",
        "CCMSLIB00000001567",
        "CCMSLIB00000001577",
        "CCMSLIB00000001587",
        "CCMSLIB00000001597",
        "CCMSLIB00000001607",
        "CCMSLIB00000001617",
        "CCMSLIB00000001627",
        "CCMSLIB00000001637",
    ]

    spectrum_data = FSInputDataGateway().load_spectrum_ids(
        local_gnps_small_json, spectrum_ids
    )

    assert len(spectrum_data) == len(spectrum_ids)
    assert set(spectrum_ids) == {d["SpectrumID"] for d in spectrum_data}
