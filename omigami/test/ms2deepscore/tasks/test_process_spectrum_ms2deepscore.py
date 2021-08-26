import os

import pytest
from mock import MagicMock
from prefect import Flow

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.tasks.process_spectrum import (
    ProcessSpectrumParameters,
    ProcessSpectrum,
)
from omigami.test.conftest import ASSETS_DIR


def test_process_spectrum_calls(spectrum_ids, common_cleaned_data):
    fs_gtw = MagicMock(spec=FSDataGateway)
    spectrum_gtw = MagicMock(spec=MS2DeepScoreRedisSpectrumDataGateway)
    spectrum_gtw.read_spectra.return_value = common_cleaned_data
    parameters = ProcessSpectrumParameters("some-path", "positive")

    with Flow("test-flow") as test_flow:
        ProcessSpectrum(fs_gtw, spectrum_gtw, parameters)(spectrum_ids)

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
    tmpdir,
):
    fs_gtw = FSDataGateway()
    spectrum_gtw = MS2DeepScoreRedisSpectrumDataGateway()
    spectrum_binner_output_path = str(tmpdir / "spectrum_binner.pkl")
    ion_mode = "positive"
    parameters = ProcessSpectrumParameters(spectrum_binner_output_path, ion_mode)
    spectrum_ids_chunks = [
        spectrum_ids[x : x + 10] for x in range(0, len(spectrum_ids), 10)
    ]
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(fs_gtw, spectrum_gtw, parameters)(
            spectrum_ids_chunks
        )

    res = test_flow.run()
    res_spectrum_ids = res.result[process_task].result

    assert res_spectrum_ids
    assert not spectrum_gtw.list_missing_binned_spectra(res_spectrum_ids, ion_mode)
    assert os.path.exists(spectrum_binner_output_path)
