import datetime
from typing import List

from gensim.models import Word2Vec
from prefect import task
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model


class ModelTrainer:
    def __init__(self):
        pass

    @staticmethod
    def train_model(
        documents: List[SpectrumDocument],
        iterations: int = None,
        window: int = None,
    ) -> Word2Vec:
        """
        Parameters
        ----------
        documents: List[SpectrumDocument]
            List of documents, each document being a SpectrumDocument that has a words attribute
        iterations: List[int]
            Specifies the number of training iterations.
        window: int
            Window size for context words (small for local context, larger for global context).
            Spec2Vec expects large windows.
        Returns
        -------
        model: Word2Vec
            A trained Word2Vec model
        """
        model = train_new_word2vec_model(
            [d.words for d in documents], iterations=iterations, window=window
        )
        return model


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def train_model_task(
    documents: List[SpectrumDocument],
    n_decimals: int,
    iterations: int = None,
    window: int = None,
) -> Word2Vec:
    model_trainer = ModelTrainer()
    model = model_trainer.train_model(documents, iterations, window)
    return model
