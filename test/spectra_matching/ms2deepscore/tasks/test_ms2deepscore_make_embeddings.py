import os

import pytest
from prefect import Flow

from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.spectra_matching.ms2deepscore.storage.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.ms2deepscore.tasks import MakeEmbeddings
from test.spectra_matching.conftest import ASSETS_DIR


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
def test_make_embeddings(binned_spectra_stored, binned_spectra, siamese_model_path):
    spectrum_ids = [spectrum.get("spectrum_id") for spectrum in binned_spectra]
    spectrum_gtw = MS2DeepScoreRedisSpectrumDataGateway()
    fs_gtw = MS2DeepScoreFSDataGateway()
    with Flow("test") as flow:
        MakeEmbeddings(spectrum_gtw, fs_gtw, "positive")(
            {"ms2deepscore_model_path": siamese_model_path},
            "1",
            spectrum_ids,
        )

    state = flow.run()
    assert state.is_successful()
    assert len(spectrum_gtw.read_embeddings("positive", spectrum_ids)) == len(
        spectrum_ids
    )
