from unittest.mock import Mock


from omigami.spectra_matching.ms2deepscore.helper_classes.siamese_model_trainer import (
    SiameseModelTrainer,
    SplitRatio,
)
from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.spectra_matching.ms2deepscore.storage.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)


def test_train_validation_test_split(binned_spectra_to_train, tanimoto_scores):
    split_ratio = SplitRatio(0.8, 0.1, 0.1)
    trainer = SiameseModelTrainer(
        Mock(MS2DeepScoreRedisSpectrumDataGateway), "positive", split_ratio=split_ratio
    )
    generators = trainer._train_validation_test_split(
        binned_spectra_to_train, tanimoto_scores, 100
    )
    n_inchikeys = len(tanimoto_scores)

    train_inchikeys = set(
        [
            spectrum.get("inchikey")[:14]
            for spectrum in generators["training"].binned_spectrums
        ]
    )
    validation_inchikeys = set(
        [
            spectrum.get("inchikey")[:14]
            for spectrum in generators["validation"].binned_spectrums
        ]
    )
    test_inchikeys = set(
        [
            spectrum.get("inchikey")[:14]
            for spectrum in generators["testing"].binned_spectrums
        ]
    )

    n_train = int(split_ratio.train * n_inchikeys)
    n_validation = int(split_ratio.validation * n_inchikeys)
    n_test = n_inchikeys - n_validation - n_train
    assert len(train_inchikeys) == n_train
    assert len(validation_inchikeys) == n_validation
    assert len(test_inchikeys) == n_test


def test_train_model(
    binned_spectra_to_train_path,
    tanimoto_scores_path,
    fitted_spectrum_binner,
    binned_spectra_to_train,
):
    layer_base_dims = (600, 500, 400)
    split_ratio = SplitRatio(0.6, 0.2, 0.2)
    trainer = SiameseModelTrainer(
        MS2DeepScoreFSDataGateway(),
        binned_spectra_to_train_path,
        epochs=5,
        split_ratio=split_ratio,
    )
    model = trainer.train(tanimoto_scores_path, fitted_spectrum_binner)

    assert len(model.model.layers) == len(layer_base_dims) + 1
    assert model.input_dim == len(fitted_spectrum_binner.known_bins)
