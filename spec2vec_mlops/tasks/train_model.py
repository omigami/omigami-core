import datetime
from typing import List

from gensim.models import Word2Vec
from prefect import task
from spec2vec import SpectrumDocument

from spec2vec_mlops.helper_classes.model_trainer import ModelTrainer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def train_model_task(
    documents: List[SpectrumDocument],
    iterations: int = None,
    window: int = None,
) -> Word2Vec:
    model_trainer = ModelTrainer()
    model = model_trainer.train_model(documents, iterations, window)
    return model
