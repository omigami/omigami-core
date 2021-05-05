from unittest.mock import MagicMock

from drfs import DRPath
from drfs.filesystems import get_fs
from prefect import Flow
from prefect.engine.results import LocalResult

from spec2vec_mlops import config
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway
from spec2vec_mlops.tasks import DownloadData
from spec2vec_mlops.tasks.data_gateway import InputDataGateway

SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]


def test_download_data():
    input_dgw = MagicMock(spec=InputDataGateway)
    input_dgw.download_gnps.return_value = "download"
    with Flow("test-flow") as test_flow:
        download = DownloadData(input_dgw)()

    res = test_flow.run()

    assert res.is_successful()
    assert res.result[download].result == "download"
    input_dgw.download_gnps.assert_called_once()


def test_download_data_twice(tmpdir):
    fs = get_fs(tmpdir)
    input_dgw = FSInputDataGateway()
    ds_dir = DRPath(tmpdir) / "datasets"
    result = LocalResult(dir=ds_dir)
    with Flow("test-flow") as test_flow:
        download = DownloadData(input_dgw, result, "gnps.json")(
            SOURCE_URI_PARTIAL_GNPS, str(ds_dir)
        )

    res = test_flow.run()
    res_2 = test_flow.run()

    assert res.is_successful()
    assert fs.exists(DRPath(tmpdir) / "datasets/gnps.json")
    assert res_2.result[download].is_cached()
