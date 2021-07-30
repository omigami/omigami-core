from dataclasses import dataclass
from logging import Logger
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from ms2deepscore import SpectrumBinner, BinnedSpectrum
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
from ms2deepscore.models import SiameseModel
from tensorflow import keras

from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway


@dataclass
class SplitRatio:
    train: float = 0.9
    validation: float = 0.05
    test: float = 0.05


class SiameseModelTrainer:
    def __init__(
        self,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        epochs: int = 50,
        learning_rate: float = 0.001,
        layer_base_dims: Tuple[int] = (600, 500, 400),
        embedding_dim: int = 400,
        dropout_rate: float = 0.2,
        split_ratio: SplitRatio = SplitRatio(),
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
        logger: Logger = None,
    ) -> SiameseModel:
        binned_spectra = self._spectrum_gtw.read_binned_spectra(spectrum_ids)
        tanimoto_scores = pd.read_pickle(scores_output_path, compression="gzip")

        data_generators = self._train_validation_test_split(
            binned_spectra, tanimoto_scores, len(spectrum_binner.known_bins)
        )
        if logger:
            logger.info(
                f"{len(data_generators['training'].binned_spectrums)} spectra in "
                f"training data "
            )
            logger.info(
                f"{len(data_generators['validation'].binned_spectrums)} spectra in "
                f"validation data "
            )
            logger.info(
                f"{len(data_generators['testing'].binned_spectrums)} spectra in test "
                f"data "
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
        model.fit(
            data_generators["training"],
            validation_data=data_generators["validation"],
            epochs=self._epochs,
        )

        return model

    def _train_validation_test_split(
        self,
        binned_spectra: List[BinnedSpectrum],
        tanimoto_scores: pd.DataFrame,
        input_vector_dimension: int,
    ) -> Dict[str, DataGeneratorAllSpectrums]:
        np.random.seed(100)

        n_inchikeys = len(tanimoto_scores)
        idx = np.arange(0, n_inchikeys)
        n_train = int(self._split_ratio.train * n_inchikeys)
        n_validation = int(self._split_ratio.validation * n_inchikeys)
        # TODO: n_test is not used?!?
        n_test = n_inchikeys - n_train - n_validation

        train_idx = np.random.choice(idx, n_train, replace=False)
        validation_idx = np.random.choice(
            list(set(idx) - set(train_idx)), n_validation, replace=False
        )
        test_idx = list(set(idx) - set(train_idx) - set(validation_idx))

        idxs = {
            "training": train_idx,
            "validation": validation_idx,
            "testing": test_idx,
        }
        spectra = {
            key: self._get_binned_spectra_from_inchikey_idx(
                tanimoto_scores, idx, binned_spectra
            )
            for key, idx in idxs.items()
        }

        data_generators = {
            key: DataGeneratorAllSpectrums(
                binned_spectrums=spectra_group,
                reference_scores_df=tanimoto_scores,
                dim=input_vector_dimension,
            )
            for key, spectra_group in spectra.items()
        }

        return data_generators

    @staticmethod
    def _get_binned_spectra_from_inchikey_idx(
        tanimoto_scores: pd.DataFrame,
        idx: np.array,
        binned_spectra: List[BinnedSpectrum],
    ) -> List[BinnedSpectrum]:
        inchikeys14 = tanimoto_scores.index.to_numpy()[idx]
        return [s for s in binned_spectra if s.get("inchikey")[:14] in inchikeys14]
