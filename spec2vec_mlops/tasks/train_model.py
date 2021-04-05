import datetime

from gensim.models import Word2Vec
from prefect import task

from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumIDStorer,
    DocumentStorer,
)
from spec2vec_mlops.helper_classes.model_trainer import ModelTrainer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def train_model_task(
    iterations: int = 25,
    window: int = 500,
) -> Word2Vec:
    ids_storer = SpectrumIDStorer("spectrum_ids_info")
    document_storer = DocumentStorer("document_info")
    all_spectrum_ids = ids_storer.read()
    documents = document_storer.read(all_spectrum_ids)
    model_trainer = ModelTrainer()
    model = model_trainer.train_model(documents, iterations, window)
    return model
