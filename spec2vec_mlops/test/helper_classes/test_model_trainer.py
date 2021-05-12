import os
import pickle

import gensim
import pytest
from pytest_redis import factories
from spec2vec.utils import TrainingProgressLogger

from spec2vec_mlops import config
from spec2vec_mlops.gateways.redis_gateway import RedisSpectrumDataGateway
from spec2vec_mlops.helper_classes.model_trainer import spec2vec_settings

DOCUMENT_HASHES = config["redis"]["document_hashes"]

redis_db = factories.redisdb("redis_nooproc")


@pytest.fixture()
def documents_stored(redis_db, cleaned_data, documents_data):
    pipe = redis_db.pipeline()
    for i, document in enumerate(documents_data):
        pipe.hset(
            DOCUMENT_HASHES,
            cleaned_data[i].metadata["spectrum_id"],
            pickle.dumps(document),
        )
    pipe.execute()


def test_spec2vec_settings():
    iterations = 5
    callbacks, settings = spec2vec_settings(iterations=iterations)
    assert isinstance(callbacks[0], TrainingProgressLogger)
    assert settings["iter"] == iterations


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_word2vec_training_with_iterator(documents_stored):
    callbacks, settings = spec2vec_settings(iterations=2)
    dgw = RedisSpectrumDataGateway()
    documents = dgw.read_documents_iter()

    model = gensim.models.Word2Vec(sentences=documents, callbacks=callbacks, **settings)

    assert len(documents.spectra_ids) == model.corpus_count
