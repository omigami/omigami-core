from dataclasses import dataclass
from logging import Logger
from typing import List, Dict

import numpy as np
import pandas as pd
from ms2deepscore import SpectrumBinner, BinnedSpectrum
from ms2deepscore.data_generators import DataGeneratorAllSpectrums
from ms2deepscore.models import SiameseModel
from tensorflow import keras

# NN Architecture parameters originated from MS2DS paper
from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import MS2DeepScoreFSDataGateway

SIAMESE_MODEL_PARAMS = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "layer_base_dims": (600, 500, 400),
    "embedding_dim": 400,
    "dropout_rate": 0.2,
}


@dataclass
class SplitRatio:
    train: float = 0.9
    validation: float = 0.05
    test: float = 0.05


class SiameseModelTrainer:
    def __init__(
        self,
        fs_dgw: MS2DeepScoreFSDataGateway,
        binned_spectra_path: str,
        epochs: int = 50,
        split_ratio: SplitRatio = SplitRatio(),
    ):
        self._fs_dgw = fs_dgw
        self._binned_spectra_path = binned_spectra_path
        self._epochs = epochs
        self._split_ratio = split_ratio
        self._learning_rate = SIAMESE_MODEL_PARAMS["learning_rate"]
        self._layer_base_dims = SIAMESE_MODEL_PARAMS["layer_base_dims"]
        self._embedding_dim = SIAMESE_MODEL_PARAMS["embedding_dim"]
        self._dropout_rate = SIAMESE_MODEL_PARAMS["dropout_rate"]
        self._batch_size = SIAMESE_MODEL_PARAMS["batch_size"]

    def train(
        self,
        scores_output_path: str,
        spectrum_binner: SpectrumBinner,
        logger: Logger = None,
    ) -> SiameseModel:
        binned_spectra = self._fs_dgw.read_from_file(self._binned_spectra_path)
        
        tanimoto_scores = pd.read_pickle(scores_output_path, compression="gzip")

        data_generators = self._train_validation_test_split(
            binned_spectra,
            tanimoto_scores,
            len(spectrum_binner.known_bins),
            batch_size=self._batch_size,
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
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=self._learning_rate),
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
        **kwargs,
    ) -> Dict[str, DataGeneratorAllSpectrums]:

        np.random.seed(100)

        n_inchikeys = len(tanimoto_scores)
        idx = np.arange(0, n_inchikeys)
        n_train = int(self._split_ratio.train * n_inchikeys)
        n_validation = int(self._split_ratio.validation * n_inchikeys)
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
                **kwargs,
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
