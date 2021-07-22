from typing import List, Tuple

import numpy as np
import pandas as pd
from ms2deepscore import SpectrumBinner, BinnedSpectrum
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
from ms2deepscore.models import SiameseModel
from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from tensorflow import keras


class SiameseModelTrainer:
    def __init__(
        self,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        epochs: int = 50,
        learning_rate: float = 0.001,
        layer_base_dims: Tuple[int] = (600, 500, 400),
        embedding_dim: int = 400,
        dropout_rate: float = 0.2,
        test_size: float = 0.2,
        input_vector_dimension: int = 100,
    ):
        self._spectrum_gtw = spectrum_dgw
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._layer_base_dims = layer_base_dims
        self._embedding_dim = embedding_dim
        self._dropout_rate = dropout_rate
        self._test_size = test_size
        self._input_vector_dimension = input_vector_dimension

    def train(
        self,
        spectrum_ids: List[str],
        scores_output_path: str,
        spectrum_binner: SpectrumBinner,
    ):
        binned_spectra = self._spectrum_gtw.read_binned_spectra(spectrum_ids)
        tanimoto_scores = pd.read_pickle(scores_output_path, compression="gzip")

        train_data_gen, test_data_gen = self._train_test_split(
            binned_spectra, tanimoto_scores
        )
        model = SiameseModel(
            spectrum_binner,
            base_dims=self._layer_base_dims,
            embedding_dim=self._embedding_dim,
            dropout_rate=self._dropout_rate,
        )
        model.compile(
            loss="mse", optimizer=keras.optimizers.Adam(lr=self._learning_rate)
        )
        # model.summary()
        model.fit(train_data_gen, validation_data=test_data_gen, epochs=self._epochs)

        return model

    def _train_test_split(
        self, binned_spectra: List[BinnedSpectrum], tanimoto_scores: pd.DataFrame
    ) -> Tuple[DataGeneratorAllSpectrums, DataGeneratorAllSpectrums]:
        mask = np.random.permutation(len(binned_spectra))
        n_train = round((1 - self._test_size) * len(binned_spectra))

        training_data = np.array(binned_spectra)[mask[:n_train]]
        testing_data = np.array(binned_spectra)[mask[n_train:]]

        train_data_generator = DataGeneratorAllSpectrums(
            binned_spectrums=training_data,
            reference_scores_df=tanimoto_scores,
            dim=self._input_vector_dimension,
        )
        test_data_generator = DataGeneratorAllSpectrums(
            binned_spectrums=testing_data,
            reference_scores_df=tanimoto_scores,
            dim=self._input_vector_dimension,
        )

        return train_data_generator, test_data_generator
