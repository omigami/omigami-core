from unittest.mock import Mock

from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.ms2deepscore.helper_classes.siamese_model_trainer import (
    SiameseModelTrainer,
)


def test_train_validation_test_split(binned_spectra, binned_spectra_tanimoto_score):
    split_ratio = (0.8, 0.1, 0.1)
    trainer = SiameseModelTrainer(
        Mock(MS2DeepScoreRedisSpectrumDataGateway), split_ratio=split_ratio
    )
    train, validation, test = trainer._train_validation_test_split(
        binned_spectra, binned_spectra_tanimoto_score, 100
    )
    n_inchikeys = len(binned_spectra_tanimoto_score)

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


def test_train_model(
    binned_spectra_stored,
    spectrum_ids,
    binned_spectra_tanimoto_score_path,
    fitted_spectrum_binner,
):
    trainer = SiameseModelTrainer(MS2DeepScoreRedisSpectrumDataGateway())
    trainer.train(
        spectrum_ids, binned_spectra_tanimoto_score_path, fitted_spectrum_binner
    )
