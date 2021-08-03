import os

import pytest
from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.ms2deepscore.tasks.make_embeddings import (
    MakeEmbeddings,
    MakeEmbeddingsParameters,
)
from omigami.test.conftest import ASSETS_DIR
from prefect import Flow


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
def test_make_embeddings(
    binned_spectra_stored, binned_spectra, ms2deepscore_real_model_path
):
    spectrum_ids = [spectrum.get("spectrum_id") for spectrum in binned_spectra]
    parameters = MakeEmbeddingsParameters(ion_mode="positive")
    spectrum_gtw = MS2DeepScoreRedisSpectrumDataGateway()
    with Flow("test") as flow:
        MakeEmbeddings(spectrum_gtw, parameters)(
            ms2deepscore_real_model_path, {"run_id": "1"}, spectrum_ids
        )

    state = flow.run()
    assert state.is_successful()
    assert len(spectrum_gtw.read_embeddings("1", spectrum_ids)) == len(spectrum_ids)
