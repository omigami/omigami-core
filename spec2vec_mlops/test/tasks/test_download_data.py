from unittest.mock import MagicMock

from prefect import Flow

from spec2vec_mlops.tasks import DownloadData
from spec2vec_mlops.tasks.data_gateway import InputDataGateway


def test_download_data():
    input_dgw = MagicMock(spec=InputDataGateway)
    input_dgw.download_gnps.return_value = "download"
    with Flow("test-flow") as test_flow:
        download = DownloadData(input_dgw)()

    res = test_flow.run()

    assert res.is_successful()
    assert res.result[download].result == "download"
    input_dgw.download_gnps.assert_called_once()
