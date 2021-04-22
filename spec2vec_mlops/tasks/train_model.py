import datetime
from typing import Union, List

import gensim
from gensim.models import Word2Vec
from prefect import task
from spec2vec.model_building import (
    set_spec2vec_defaults,
    learning_rates_to_gensim_style,
)
from spec2vec.utils import TrainingProgressLogger, ModelSaver

from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumIDStorer,
    DocumentStorer,
)


class MyCorpus:
    """An iterator that yields spectra (lists of str)."""

    def __iter__(self):
        ids_storer = SpectrumIDStorer("spectrum_ids_info")
        document_storer = DocumentStorer("document_info")
        all_spectrum_ids = ids_storer.read_online()
        for spectrum_id in all_spectrum_ids:
            yield document_storer.read_online([spectrum_id])[0]


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def train_model_task(
    iterations: int = 25,
    window: int = 500,
) -> Word2Vec:
    callbacks, settings = spec2vec_settings(iterations=iterations, window=window)
    documents = MyCorpus()
    model = gensim.models.Word2Vec(sentences=documents, callbacks=callbacks, **settings)
    return model


def spec2vec_settings(
    iterations: Union[List[int], int],
    filename: str = None,
    progress_logger: bool = True,
    **settings
):
    settings = set_spec2vec_defaults(**settings)

    num_of_epochs = max(iterations) if isinstance(iterations, list) else iterations

    # Convert spec2vec style arguments to gensim style arguments
    settings = learning_rates_to_gensim_style(num_of_epochs, **settings)

    # Set callbacks
    callbacks = []
    if progress_logger:
        training_progress_logger = TrainingProgressLogger(num_of_epochs)
        callbacks.append(training_progress_logger)
    if filename:
        if isinstance(iterations, int):
            iterations = [iterations]
        model_saver = ModelSaver(num_of_epochs, iterations, filename)
        callbacks.append(model_saver)

    return callbacks, settings
