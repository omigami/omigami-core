from unittest.mock import MagicMock
from prefect import Flow
from omigami.data_gateway import InputDataGateway
from omigami.spec2vec.gateways.input_data_gateway import FSInputDataGateway
from omigami.ms2deepscore.tasks.download_pre_trained_model import (
    DownloadPreTrainedModelParameters,
    DownloadPreTrainedModel,
)
from omigami.test.conftest import ASSETS_DIR


def test_download_pre_trained_model(tmpdir, mock_default_config):

    download_parameters = DownloadPreTrainedModelParameters(
        model_uri="fake_uri", output_dir=tmpdir
    )
    input_dgw = MagicMock(spec=InputDataGateway)
    input_dgw.download_ms2deep_model.return_value = download_parameters.output_path

    with Flow("ms2deepscore-test-flow") as flow:
        download = DownloadPreTrainedModel(input_dgw, download_parameters)()

    res = flow.run()

    assert res.is_successful()
    assert res.result[download].result == download_parameters.output_path

    input_dgw.download_ms2deep_model.assert_called_once_with(
        download_parameters.model_uri, download_parameters.output_path
    )


def test_download_pretrained_model_existing(mock_default_config):

    input_dgw = FSInputDataGateway()
    input_dgw.download_ms2deep_model = lambda *args: None

    download_parameters = DownloadPreTrainedModelParameters(
        model_uri="fake_uri",
        output_dir=ASSETS_DIR / "ms2deepscore" / "pretrained",
        file_name="fake_pretrained_model.hdf5",
    )

    with Flow("ms2deepscore-test-flow") as flow:
        download = DownloadPreTrainedModel(input_dgw, download_parameters)()

    res = flow.run()

    assert res.is_successful()
    assert res.result[download].is_cached()
