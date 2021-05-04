from pathlib import Path

import pytest
import requests_mock
from drfs import DRPath

from spec2vec_mlops import config
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway

KEYS = config["gnps_json"]["necessary_keys"]


def test_load_gnps(local_gnps_small_json):
    for res in FSInputDataGateway().load_gnps(local_gnps_small_json):
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
    res = FSInputDataGateway().download_gnps(
        uri=uri, dataset_dir=tmpdir, dataset_id="test-ds"
    )

    assert Path(res).exists()


def test_download_and_serialize_to_remote(loaded_data, s3_mock):
    with requests_mock.Mocker() as m:
        m.get(SOURCE_URI_PARTIAL_GNPS, text="bac")
        res = FSInputDataGateway().download_gnps(
            uri=SOURCE_URI_PARTIAL_GNPS,
            dataset_dir="s3://test-bucket",
            dataset_id="test-ds",
        )
        assert DRPath(res).exists()


def test_download_already_exists(tmpdir):
    file_path = FSInputDataGateway().download_gnps(
        uri=SOURCE_URI_PARTIAL_GNPS, dataset_dir=tmpdir, dataset_id="test-ds"
    )
    same_file_path = FSInputDataGateway().download_gnps(
        uri=SOURCE_URI_PARTIAL_GNPS, dataset_dir=tmpdir, dataset_id="test-ds"
    )

    assert file_path == same_file_path


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


def test_make_path(tmpdir):
    path = FSInputDataGateway._make_path(tmpdir)

    assert isinstance(path, str)
    assert path.split(".")[1] == "json"
