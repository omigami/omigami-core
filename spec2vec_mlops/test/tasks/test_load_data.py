from unittest.mock import MagicMock

from prefect import Flow

from spec2vec_mlops.tasks import LoadData
from spec2vec_mlops.tasks.data_gateway import InputDataGateway


def test_load_data():
    gnps = list(range(50))
    input_dgw = MagicMock(spec=InputDataGateway)
    input_dgw.load_gnps.return_value = gnps
    with Flow("test-flow") as test_flow:
        load_task = LoadData(input_dgw)(chunk_size=10)

    res = test_flow.run()
    data = res.result[load_task].result

    assert res.is_successful()
    assert len(data) == 5
    assert set(data[0]) == set(range(10))
    input_dgw.load_gnps.assert_called_once()
