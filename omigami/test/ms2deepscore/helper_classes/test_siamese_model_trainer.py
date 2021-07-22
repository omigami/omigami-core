from unittest.mock import Mock

from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.ms2deepscore.helper_classes.siamese_model_trainer import (
    SiameseModelTrainer,
)


def test_train_test_split(binned_spectra, binned_spectra_tanimoto_score):
    test_size = 0.3
    trainer = SiameseModelTrainer(
        Mock(MS2DeepScoreRedisSpectrumDataGateway), test_size=test_size
    )
    train, test = trainer._train_test_split(
        binned_spectra, binned_spectra_tanimoto_score
    )

    n_train = round((1 - test_size) * len(binned_spectra))
    assert len(train.binned_spectrums) == n_train
    assert len(test.binned_spectrums) == len(binned_spectra) - n_train
