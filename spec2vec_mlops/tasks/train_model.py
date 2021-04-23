import datetime

import gensim
from gensim.models import Word2Vec
from prefect import task

from spec2vec_mlops.helper_classes.model_trainer import spec2vec_settings
from spec2vec_mlops.helper_classes.spectra_iterator import SpectraIterator
from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumIDStorer,
    DocumentStorer,
)


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def train_model_task(
    iterations: int = 25,
    window: int = 500,
) -> Word2Vec:
    ids_storer = SpectrumIDStorer("spectrum_ids_info")
    document_storer = DocumentStorer("document_info")
    all_spectrum_ids = ids_storer.read_online()

    documents = SpectraIterator(all_spectrum_ids, document_storer)

    callbacks, settings = spec2vec_settings(iterations=iterations, window=window)
    model = gensim.models.Word2Vec(sentences=documents, callbacks=callbacks, **settings)
    return model
