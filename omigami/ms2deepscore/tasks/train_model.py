from tensorflow import keras
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
from ms2deepscore.models import SiameseModel
from prefect import Task

from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)


@dataclass
class TrainModelParameters:
    epochs: int = 50
    learning_rate: float = 0.001


class TrainModel(Task):
    def __init__(
        self,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        train_paramaters: TrainModelParameters,
    ):
        self.spectrum_gtw = spectrum_dgw
        self._epochs = train_paramaters.epochs
        self._learning_rate = train_paramaters.learning_rate

    def run(self):
        binned_spectra = self.spectrum_gtw.read_binned_spectra()
        tanimoto_scores = self.spectrum_gtw.read_tanimoto_scores()
        spectrum_binner = self.spectrum_gtw.get_spectrum_binner()

        train_data_gen, test_data_gen = self._train_test_split(
            binned_spectra, tanimoto_scores
        )
        model = SiameseModel(
            spectrum_binner,
            base_dims=(600, 500, 400),
            embedding_dim=400,
            dropout_rate=0.2,
        )
        model.compile(
            loss="mse", optimizer=keras.optimizers.Adam(lr=self._learning_rate)
        )
        # model.summary()
        model.fit(train_data_gen, validation_data=test_data_gen, epochs=self._epochs)

        return model

    def _train_test_split(self, binned_spectra, tanimoto_scores):
        # TODO: make the data split

        train_data_generator = DataGeneratorAllSpectrums(
            binned_spectrums=binned_spectra,
            reference_scores_df=tanimoto_scores,
            dim=100,
        )

        test_data_generator = DataGeneratorAllSpectrums(
            binned_spectrums=binned_spectra,
            reference_scores_df=tanimoto_scores,
            dim=100,
        )

        return train_data_generator, test_data_generator
