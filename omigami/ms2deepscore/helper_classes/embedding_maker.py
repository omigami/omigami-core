import numpy as np
from ms2deepscore import BinnedSpectrum
from ms2deepscore.models import load_model as ms2deepscore_load_model
from omigami.ms2deepscore.entities.embedding import Embedding
from omigami.ms2deepscore.helper_classes.ms2deepscore_embedding import (
    MS2DeepScoreEmbedding,
)


class EmbeddingMaker:
    def make_embedding(
        self,
        model_path: str,
        binned_spectrum: BinnedSpectrum,
    ) -> Embedding:
        siamese_model = ms2deepscore_load_model(model_path)
        model = MS2DeepScoreEmbedding(siamese_model)
        vector = model.model.base.predict(
            self._create_input_vector(binned_spectrum, model.input_vector_dim)
        )

        return Embedding(
            vector,
            binned_spectrum.metadata.get("spectrum_id"),
            binned_spectrum.get("inchikey"),
        )

    def _create_input_vector(
        self, binned_spectrum: BinnedSpectrum, input_vector_dim: int
    ):
        """Creates input vector for model.base based on binned peaks and intensities"""
        X = np.zeros((1, input_vector_dim))

        idx = np.array([int(x) for x in binned_spectrum.binned_peaks.keys()])
        values = np.array(list(binned_spectrum.binned_peaks.values()))
        X[0, idx] = values
        return X
