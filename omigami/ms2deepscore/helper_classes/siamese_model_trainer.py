from logging import getLogger
from typing import List, Tuple

import numpy as np
import pandas as pd
from ms2deepscore import SpectrumBinner, BinnedSpectrum
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
from ms2deepscore.models import SiameseModel
from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from tensorflow import keras

log = getLogger(__name__)


class SiameseModelTrainer:
    def __init__(
        self,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        epochs: int = 50,
        learning_rate: float = 0.001,
        layer_base_dims: Tuple[int] = (600, 500, 400),
        embedding_dim: int = 400,
        dropout_rate: float = 0.2,
        split_ratio: Tuple[float, float, float] = (0.9, 0.05, 0.05),
    ):
        self._spectrum_gtw = spectrum_dgw
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._layer_base_dims = layer_base_dims
        self._embedding_dim = embedding_dim
        self._dropout_rate = dropout_rate
        self._split_ratio = split_ratio

    def train(
        self,
        spectrum_ids: List[str],
        scores_output_path: str,
        spectrum_binner: SpectrumBinner,
    ):
        binned_spectra = self._spectrum_gtw.read_binned_spectra(spectrum_ids)
        tanimoto_scores = pd.read_pickle(scores_output_path, compression="gzip")

        (
            train_data_gen,
            validation_data_gen,
            test_data_gen,
        ) = self._train_validation_test_split(
            binned_spectra, tanimoto_scores, len(spectrum_binner.known_bins)
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
        model.fit(
            train_data_gen, validation_data=validation_data_gen, epochs=self._epochs
        )

        return model

    def _train_validation_test_split(
        self,
        binned_spectra: List[BinnedSpectrum],
        tanimoto_scores: pd.DataFrame,
        input_vector_dimension: int,
    ) -> Tuple[
        DataGeneratorAllSpectrums, DataGeneratorAllSpectrums, DataGeneratorAllSpectrums
    ]:

        np.random.seed(100)

        n_inchikeys = len(tanimoto_scores)
        idx = np.arange(0, n_inchikeys)
        n_train = int(self._split_ratio[0] * n_inchikeys)
        n_validation = int(self._split_ratio[1] * n_inchikeys)
        n_test = n_inchikeys - n_train - n_validation

        log.info(
            f"Split dataset into train: {n_train}, validation: {n_validation } "
            f"and test: {n_test} InChiKeys"
        )

        train_idx = np.random.choice(idx, n_train, replace=False)
        validation_idx = np.random.choice(
            list(set(idx) - set(train_idx)), n_validation, replace=False
        )
        test_idx = list(set(idx) - set(train_idx) - set(validation_idx))

        spectra_training = self._get_binned_spectra_from_inchikey_idx(
            tanimoto_scores, train_idx, binned_spectra
        )
        log.info(f"{len(spectra_training)} spectra in training data")
        spectra_validation = self._get_binned_spectra_from_inchikey_idx(
            tanimoto_scores, validation_idx, binned_spectra
        )
        log.info(f"{len(spectra_validation)} spectra in validation data")
        spectra_testing = self._get_binned_spectra_from_inchikey_idx(
            tanimoto_scores, test_idx, binned_spectra
        )
        log.info(f"{len(spectra_testing)} spectra in test data")

        train_data_generator = DataGeneratorAllSpectrums(
            binned_spectrums=spectra_training,
            reference_scores_df=tanimoto_scores,
            dim=input_vector_dimension,
        )
        validation_data_generator = DataGeneratorAllSpectrums(
            binned_spectrums=spectra_validation,
            reference_scores_df=tanimoto_scores,
            dim=input_vector_dimension,
        )
        testing_data_generator = DataGeneratorAllSpectrums(
            binned_spectrums=spectra_testing,
            reference_scores_df=tanimoto_scores,
            dim=input_vector_dimension,
        )

        return train_data_generator, validation_data_generator, testing_data_generator

    @staticmethod
    def _get_binned_spectra_from_inchikey_idx(
        tanimoto_scores: pd.DataFrame,
        idx: np.array,
        binned_spectra: List[BinnedSpectrum],
    ):
        inchikeys14 = tanimoto_scores.index.to_numpy()[idx]
        return [s for s in binned_spectra if s.get("inchikey")[:14] in inchikeys14]
