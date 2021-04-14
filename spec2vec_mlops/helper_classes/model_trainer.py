from typing import List, Union

from gensim.models import Word2Vec
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model

from spec2vec_mlops.entities.feast_spectrum_document import FeastSpectrumDocument


class ModelTrainer:
    @staticmethod
    def train_model(
        documents: List[Union[SpectrumDocument, FeastSpectrumDocument]],
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
            documents, iterations=iterations, window=window
        )
        return model
