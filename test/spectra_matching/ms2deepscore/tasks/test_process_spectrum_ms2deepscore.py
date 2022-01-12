import os
from unittest.mock import MagicMock

import pytest
from prefect import Flow

from omigami.spectra_matching.ms2deepscore.storage import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.ms2deepscore.tasks import (
    ProcessSpectrumParameters,
    ProcessSpectrum,
)
from omigami.spectra_matching.storage import FSDataGateway
from test.spectra_matching.conftest import ASSETS_DIR


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
    siamese_model_path,
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
