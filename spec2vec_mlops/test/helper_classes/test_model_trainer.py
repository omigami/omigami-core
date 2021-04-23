from spec2vec.utils import TrainingProgressLogger

from spec2vec_mlops.helper_classes.model_trainer import spec2vec_settings


def test_spec2vec_settings():
    iterations = 5
    callbacks, settings = spec2vec_settings(iterations=iterations)
    assert isinstance(callbacks[0], TrainingProgressLogger)
    assert settings["iter"] == iterations
