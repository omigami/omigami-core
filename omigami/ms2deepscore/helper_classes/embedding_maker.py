import numpy as np
from ms2deepscore import BinnedSpectrum

from omigami.ms2deepscore.helper_classes.ms2deepscore_embedding import (
    MS2DeepScoreEmbedding,
)
from omigami.ms2deepscore.entities.embedding import Embedding


class EmbeddingMaker:
    def make_embedding(
        self,
        model: MS2DeepScoreEmbedding,
        binned_spectrum: BinnedSpectrum,
    ) -> Embedding:
        vector = model.model.base.predict(
            self._create_input_vector(binned_spectrum, model.input_vector_dim)
        )

        return Embedding(
            vector=vector,
            spectrum_id=binned_spectrum.metadata.get("spectrum_id"),
            inchikey=binned_spectrum.get("inchikey"),
        )

    @staticmethod
    def _create_input_vector(
        binned_spectrum: BinnedSpectrum, input_vector_dim: int
    ) -> np.ndarray:
        """Creates input vector for model.base based on binned peaks and intensities"""
        X = np.zeros((1, input_vector_dim))

        idx = np.array([int(x) for x in binned_spectrum.binned_peaks.keys()])
        values = np.array(list(binned_spectrum.binned_peaks.values()))
        X[0, idx] = values
        return X
