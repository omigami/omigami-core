import os
from unittest.mock import Mock

import pytest

from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.ms2deepscore.helper_classes.siamese_model_trainer import (
    SiameseModelTrainer,
)


def test_train_validation_test_split(binned_spectra_to_train, tanimoto_scores):
    split_ratio = (0.8, 0.1, 0.1)
    trainer = SiameseModelTrainer(
        Mock(MS2DeepScoreRedisSpectrumDataGateway), split_ratio=split_ratio
    )
    train, validation, test = trainer._train_validation_test_split(
        binned_spectra_to_train, tanimoto_scores, 100
    )
    n_inchikeys = len(tanimoto_scores)

    train_inchikeys = set(
        [spectrum.get("inchikey")[:14] for spectrum in train.binned_spectrums]
    )
    validation_inchikeys = set(
        [spectrum.get("inchikey")[:14] for spectrum in validation.binned_spectrums]
    )
    test_inchikeys = set(
        [spectrum.get("inchikey")[:14] for spectrum in test.binned_spectrums]
    )

    n_train = int(split_ratio[0] * n_inchikeys)
    n_validation = int(split_ratio[1] * n_inchikeys)
    n_test = n_inchikeys - n_validation - n_train
    assert len(train_inchikeys) == n_train
    assert len(validation_inchikeys) == n_validation
    assert len(test_inchikeys) == n_test


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_train_model(
    binned_spectra_to_train_stored,
    tanimoto_scores_path,
    fitted_spectrum_binner,
    binned_spectra_to_train,
):
    layer_base_dims = (600, 500, 400)
    trainer = SiameseModelTrainer(MS2DeepScoreRedisSpectrumDataGateway(), epochs=5)
    model = trainer.train([], tanimoto_scores_path, fitted_spectrum_binner)

    assert len(model.model.layers) == len(layer_base_dims) + 1
    assert model.input_dim == len(fitted_spectrum_binner.known_bins)
