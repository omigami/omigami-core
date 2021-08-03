from ms2deepscore import BinnedSpectrum
from ms2deepscore.models import load_model as ms2deepscore_load_model
from omigami.ms2deepscore.entities.embedding import Embedding
from omigami.ms2deepscore.helper_classes.ms2deepscore_binned_spectrum import (
    MS2DeepScoreBinnedSpectrum,
)


class EmbeddingMaker:
    def make_embedding(
        self,
        model_path: str,
        binned_spectrum: BinnedSpectrum,
    ) -> Embedding:
        siamese_model = ms2deepscore_load_model(model_path)
        model = MS2DeepScoreBinnedSpectrum(siamese_model)
        vector = model.model.base.predict(model.create_input_vector(binned_spectrum))

        return Embedding(
            vector,
            binned_spectrum.metadata.get("spectrum_id"),
            binned_spectrum.get("inchikey"),
        )
