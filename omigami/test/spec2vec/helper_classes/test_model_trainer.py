from unittest.mock import Mock

import gensim
from pytest_redis import factories

from omigami.gateways import RedisSpectrumDataGateway
from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.gateways import Spec2VecFSDocumentDataGateway
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


def test_word2vec_training_with_iterator(save_documents, documents_directory):
    dgw = Spec2VecFSDocumentDataGateway()
    train_model_params = TrainModelParameters(2, 10)
    train_model = TrainModel(dgw, train_model_params)
    callbacks, settings = train_model._create_spec2vec_settings(epochs=2, window=10)
    documents = dgw.read_from_file(f"{documents_directory}/test.pkl")

    model = gensim.models.Word2Vec(sentences=documents, callbacks=callbacks, **settings)

    assert len(documents) == model.corpus_count


def test_load_all_documents(save_documents, documents_directory):
    gtw = Spec2VecFSDocumentDataGateway()
    train_model_params = TrainModelParameters(epochs=50, window=10)
    train_model = TrainModel(gtw, train_model_params)

    documents = train_model._load_all_document_files(
        [f"{documents_directory}/test.pkl"]
    )

    assert len(documents) == 100
