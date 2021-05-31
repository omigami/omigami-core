import datetime

import gensim
import prefect
from gensim.models import Word2Vec
from prefect import task

from omigami.helper_classes.model_trainer import spec2vec_settings
from omigami.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway
from omigami.tasks.config import DEFAULT_CONFIG


@task(**DEFAULT_CONFIG)
def train_model_task(
    iterations: int = 25,
    window: int = 500,
) -> Word2Vec:
    logger = prefect.context.get("logger")
    beg = datetime.datetime.now()

    dgw = RedisSpectrumDataGateway()
    documents = dgw.read_documents_iter()
    callbacks, settings = spec2vec_settings(iterations=iterations, window=window)
    model = gensim.models.Word2Vec(sentences=documents, callbacks=callbacks, **settings)
    logger.info(f"Train model in {datetime.datetime.now() - beg} hours.")

    return model
