import os

import gensim
import pytest
from pytest_redis import factories
from spec2vec.utils import TrainingProgressLogger

from omigami.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway
from omigami.tasks.train_model import TrainModel

redis_db = factories.redisdb("redis_nooproc")


def test_spec2vec_settings():
    epochs = 5
    callbacks, settings = TrainModel._create_spec2vec_settings(epochs=epochs, window=10)
    assert isinstance(callbacks[0], TrainingProgressLogger)
    assert settings["iter"] == epochs


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_word2vec_training_with_iterator(documents_stored):
    callbacks, settings = TrainModel._create_spec2vec_settings(epochs=2, window=10)
    dgw = RedisSpectrumDataGateway()
    documents = dgw.read_documents_iter()

    model = gensim.models.Word2Vec(sentences=documents, callbacks=callbacks, **settings)

    assert len(documents.spectrum_ids) == model.corpus_count
