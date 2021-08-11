import os
from unittest.mock import Mock

import gensim
import pytest
from pytest_redis import factories

from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)
from omigami.spec2vec.helper_classes.train_logger import (
    CustomTrainingProgressLogger,
)
from omigami.spec2vec.tasks.train_model import TrainModel, TrainModelParameters

redis_db = factories.redisdb("redis_nooproc")


def test_spec2vec_settings():
    epochs = 5
    gtw = Mock(spec=Spec2VecRedisSpectrumDataGateway)
    train_model_params = TrainModelParameters(epochs, 10)
    train_model = TrainModel(gtw, train_model_params)

    callbacks, settings = train_model._create_spec2vec_settings(
        epochs=epochs, window=10
    )

    assert isinstance(callbacks[0], CustomTrainingProgressLogger)
    assert settings["iter"] == epochs


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_word2vec_training_with_iterator(documents_stored, tmpdir, write_documents):
    dgw = Spec2VecRedisSpectrumDataGateway()
    train_model_params = TrainModelParameters(2, 10)
    train_model = TrainModel(dgw, train_model_params)
    callbacks, settings = train_model._create_spec2vec_settings(epochs=2, window=10)
    documents = dgw.read_documents(f"{tmpdir}/documents/test.pckl")

    model = gensim.models.Word2Vec(sentences=documents, callbacks=callbacks, **settings)

    assert len(documents) == model.corpus_count
