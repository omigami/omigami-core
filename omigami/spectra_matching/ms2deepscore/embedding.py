from dataclasses import dataclass

import numpy as np
from ms2deepscore import BinnedSpectrum
from ms2deepscore.models import SiameseModel

from omigami.spectra_matching.entities.embedding import Embedding


class MS2DeepScoreEmbedding:
    def __init__(self, vector:  np.ndarray, spectrum_id: str, inchikey: str):
        self.vector = vector
        self.spectrum_id = spectrum_id
        self.inchikey = inchikey


class EmbeddingMaker:
    def make_embedding(
        self, model: SiameseModel, binned_spectrum: BinnedSpectrum
    ) -> MS2DeepScoreEmbedding:
        vector = model.base.predict(
            self._create_input_vector(binned_spectrum, model.input_dim)
        )

        return MS2DeepScoreEmbedding(
            vector=vector,
            spectrum_id=binned_spectrum.metadata.get("spectrum_id"),
            inchikey=binned_spectrum.get("inchikey"),
        )

    @staticmethod
    def _create_input_vector(
        binned_spectrum: BinnedSpectrum, input_vector_dim: int
    ) -> np.ndarray:
        """
        Creates input vector for model.base based on binned peaks and intensities.
        If you refactor this method please also refactor tge same function in
        `omigami/test/ms2deepscore/test_scripts_to_update_assets.py`
        """
        X = np.zeros((1, input_vector_dim))

        idx = np.array([int(x) for x in binned_spectrum.binned_peaks.keys()])
        values = np.array(list(binned_spectrum.binned_peaks.values()))
        X[0, idx] = values
        return X
