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
    feast_core_url: str,
    iterations: int = 25,
    window: int = 500,
) -> Word2Vec:
    spectrum_ids_storer = SpectrumIDStorer(feast_core_url)
    document_storer = DocumentStorer(feast_core_url)
    all_spectrum_ids = spectrum_ids_storer.read()
    documents = document_storer.read(all_spectrum_ids)
    model_trainer = ModelTrainer()
    model = model_trainer.train_model(documents, iterations, window)
    return model
