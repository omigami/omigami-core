from typing import Union, List

from spec2vec.model_building import (
    set_spec2vec_defaults,
    learning_rates_to_gensim_style,
)
from spec2vec.utils import TrainingProgressLogger


def spec2vec_settings(iterations: Union[List[int], int], **settings):
    settings = set_spec2vec_defaults(**settings)

    num_of_epochs = max(iterations) if isinstance(iterations, list) else iterations

    # Convert spec2vec style arguments to gensim style arguments
    settings = learning_rates_to_gensim_style(num_of_epochs, **settings)

    # Set callbacks
    callbacks = []
    training_progress_logger = TrainingProgressLogger(num_of_epochs)
    callbacks.append(training_progress_logger)

    return callbacks, settings
