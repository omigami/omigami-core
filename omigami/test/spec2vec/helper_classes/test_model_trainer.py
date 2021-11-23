import os
from unittest.mock import Mock

import gensim
import pytest
from drfs.filesystems import get_fs
from pytest_redis import factories

from omigami.base.gateways import RedisSpectrumDataGateway
from omigami.base.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.gateways.fs_document_iterator import FileSystemDocumentIterator
from omigami.spec2vec.helper_classes.train_logger import (
    CustomTrainingProgressLogger,
)
from omigami.spec2vec.tasks.train_model import TrainModel, TrainModelParameters

redis_db = factories.redisdb("redis_nooproc")


def test_spec2vec_settings():
    epochs = 5
    gtw = Mock(spec=RedisSpectrumDataGateway)
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
def test_word2vec_training_with_iterator(
    documents_stored, s3_documents_directory, s3_mock
):
    dgw = FSDataGateway()

    fs = get_fs(s3_documents_directory)
    fs.makedirs(s3_documents_directory)

    train_model_params = TrainModelParameters(2, 10)
    train_model = TrainModel(dgw, train_model_params)
    callbacks, settings = train_model._create_spec2vec_settings(epochs=2, window=10)

    documents = FileSystemDocumentIterator(dgw, fs.ls(s3_documents_directory))

    model = gensim.models.Word2Vec(sentences=documents, callbacks=callbacks, **settings)

    assert len(documents) == model.corpus_count
