from unittest.mock import MagicMock

from drfs import DRPath
from drfs.filesystems import get_fs
from prefect import Flow
from prefect.engine.results import LocalResult
from prefect.engine.serializers import JSONSerializer

from spec2vec_mlops import config
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway
from spec2vec_mlops.tasks import DownloadData
from spec2vec_mlops.tasks.data_gateway import InputDataGateway
from spec2vec_mlops.test.conftest import ASSETS_DIR

SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]


def test_download_data():
    input_dgw = MagicMock(spec=InputDataGateway)
    input_dgw.download_gnps.return_value = "download"
    input_dgw.load_gnps.return_value = "gnps"
    with Flow("test-flow") as test_flow:
        download = DownloadData(input_dgw, LocalResult(), "target")()

    res = test_flow.run()

    assert res.is_successful()
    assert res.result[download].result == "gnps"
    input_dgw.download_gnps.assert_called_once_with(None, None)
    input_dgw.load_gnps.assert_called_once_with("download")


def test_download_existing_data():
    file_name = "SMALL_GNPS.json"
    input_dgw = FSInputDataGateway()
    input_dgw.download_gnps = MagicMock()
    fs = get_fs(ASSETS_DIR)
    result = LocalResult(dir=ASSETS_DIR, serializer=JSONSerializer())

    with Flow("test-flow") as test_flow:
        download = DownloadData(input_dgw, result, file_name)(
            SOURCE_URI_PARTIAL_GNPS, str(ASSETS_DIR)
        )

    res = test_flow.run()

    assert res.is_successful()
    assert fs.exists(DRPath(ASSETS_DIR) / file_name)
    assert res.result[download].is_cached()
