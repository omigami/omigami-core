import os

import numpy as np
import pytest

from omigami.spectra_matching.ms2deepscore.embedding import (
    MS2DeepScoreEmbedding,
    EmbeddingMaker,
)
from omigami.test.conftest import ASSETS_DIR


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
def test_make_embedding(binned_spectra, siamese_model):
    maker = EmbeddingMaker()
    embedding = maker.make_embedding(siamese_model, binned_spectra[0])
    assert isinstance(embedding, MS2DeepScoreEmbedding)
    assert isinstance(embedding.vector, np.ndarray)
