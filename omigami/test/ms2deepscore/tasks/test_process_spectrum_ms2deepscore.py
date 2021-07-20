import os

import pytest
from mock import MagicMock
from prefect import Flow

from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.tasks.process_spectrum import (
    ProcessSpectrumParameters,
    ProcessSpectrum,
)
from omigami.test.conftest import ASSETS_DIR


def test_process_spectrum_calls(spectrum_ids, basic_cleaned_data):
    spectrum_gtw = MagicMock(spec=MS2DeepScoreRedisSpectrumDataGateway)
    spectrum_gtw.read_spectra.return_value = basic_cleaned_data
    parameters = ProcessSpectrumParameters(spectrum_gtw, True)

    with Flow("test-flow") as test_flow:
        ProcessSpectrum(parameters)(spectrum_ids)

    res = test_flow.run()

    assert res.is_successful()
    spectrum_gtw.list_missing_binned_spectra.assert_not_called()
    spectrum_gtw.write_binned_spectra.assert_called_once()


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
@pytest.mark.skipif(
    not os.path.exists(
        str(
            ASSETS_DIR
            / "ms2deepscore"
            / "pretrained"
            / "MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5"
        )
    ),
    reason="MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5 is git ignored. Please "
    "download it from https://zenodo.org/record/4699356#.YNyD-2ZKhcA",
)
def test_process_spectrum(
    spectra_stored,
    spectrum_ids,
    ms2deepscore_real_model_path,
    mock_default_config,
):
    spectrum_gtw = MS2DeepScoreRedisSpectrumDataGateway()
    parameters = ProcessSpectrumParameters(spectrum_gtw, False)
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(parameters)(spectrum_ids)

    res = test_flow.run()
    res_spectrum_ids = res.result[process_task].result

    assert res_spectrum_ids
    assert not spectrum_gtw.list_missing_binned_spectra(res_spectrum_ids)
