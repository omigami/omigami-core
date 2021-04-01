import datetime

from gensim.models import Word2Vec
from prefect import task

from spec2vec_mlops.helper_classes.feature_loader import FeatureLoader
from spec2vec_mlops.helper_classes.model_trainer import ModelTrainer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def train_model_task(
    feast_core_url: str,
    iterations: int = 25,
    window: int = 500,
) -> Word2Vec:
    feature_loader = FeatureLoader(feast_core_url)
    all_spectrum_ids = feature_loader.load_all_spectrum_ids()
    documents = feature_loader.load_documents(spectrum_ids=all_spectrum_ids)
    model_trainer = ModelTrainer()
    model = model_trainer.train_model(documents, iterations, window)
    return model
