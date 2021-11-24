import os

import numpy as np
import pytest

from omigami.ms2deepscore.helper_classes.embedding_maker import EmbeddingMaker
from omigami.spectra_matching.entities.embedding import Embedding
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
def test_make_embedding(binned_spectra, ms2deepscore_real_predictor):
    maker = EmbeddingMaker()
    embedding = maker.make_embedding(
        ms2deepscore_real_predictor.model, binned_spectra[0]
    )
    assert isinstance(embedding, Embedding)
    assert isinstance(embedding.vector, np.ndarray)
