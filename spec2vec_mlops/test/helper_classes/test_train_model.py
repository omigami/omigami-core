from gensim.models import Word2Vec
from spec2vec_mlops.tasks.train_model import ModelTrainer


def test_train_model(documents_data):
    model_trainer = ModelTrainer()
    model = model_trainer.train_model(documents_data, iterations=10, window=5)
    assert isinstance(model, Word2Vec)
    assert model.iter == 10
    assert model.window == 5
