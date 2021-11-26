import pickle
from pathlib import Path

import numpy as np
import pytest

from omigami.ms2deepscore.entities.embedding import MS2DeepScoreEmbedding
from omigami.test.conftest import ASSETS_DIR


def create_input_vector(binned_spectrum, input_vector_dim) -> np.ndarray:
    """
    Creates input vector for model.base based on binned peaks and intensities
    Function copied from omigami/ms2deepscore/helper_classes/embedding_maker.py

    """
    X = np.zeros((1, input_vector_dim))
    idx = np.array([int(x) for x in binned_spectrum.binned_peaks.keys()])
    values = np.array(list(binned_spectrum.binned_peaks.values()))
    X[0, idx] = values
    return X


@pytest.mark.skip(
    reason="This test should only be run if "
    "`omigami/test/assets/ms2deepscore/SMALL_GNPS_as_embeddings.pkl`"
    "needs to be updated. "
)
def test_create_embeddings_from_real_predictor(
    binned_spectra, ms2deepscore_real_predictor
):
    """
    This test is to create/update `omigami/test/assets/ms2deepscore/SMALL_GNPS_as_embeddings.pkl`
    with `MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5 model`. E.g. if you rename
    `MS2DeepScoreEmbedding` from `omigami/ms2deepscore/entities/embedding.py`, it is
    necessary to run this test.

    It also requires ms2deepscore model trained on all GNPS to be available in
    `omigami/test/assets/ms2deepscore/pretrained`. Before running this, please download
    the model from https://zenodo.org/record/4699356#.YNyD-2ZKhcA, and place it in
    `omigami/test/assets/ms2deepscore/pretrained`.

    Parameters
    ----------
    binned_spectra: List[BinnedSpectrum]
    ms2deepscore_real_predictor: ms2deepscore model trained on all GNPS

    Returns
    -------
    List[MS2DeepScoreEmbedding]

    """
    path = Path(ASSETS_DIR / "ms2deepscore" / "SMALL_GNPS_as_embeddings.pkl")

    embeddings = []
    for binned_spectrum in binned_spectra:
        vector = ms2deepscore_real_predictor.model.model.base.predict(
            create_input_vector(
                binned_spectrum, ms2deepscore_real_predictor.model.input_vector_dim
            )
        )
        embeddings.append(
            MS2DeepScoreEmbedding(
                vector=vector,
                spectrum_id=binned_spectrum.metadata.get("spectrum_id"),
                inchikey=binned_spectrum.get("inchikey"),
            )
        )
    with open(str(path), "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings
