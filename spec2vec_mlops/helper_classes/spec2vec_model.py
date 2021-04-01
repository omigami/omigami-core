from typing import List, Dict, Union

from gensim.models import Word2Vec
from mlflow.pyfunc import PythonModel

from spec2vec_mlops.helper_classes.data_loader import DataLoader
from spec2vec_mlops.helper_classes.data_cleaner import DataCleaner
from spec2vec_mlops.helper_classes.document_converter import DocumentConverter
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker


class Model(PythonModel):
    def __init__(
        self,
        model: Word2Vec,
        n_decimals: int,
        intensity_weighting_power: Union[float, int],
        allowed_missing_percentage: Union[float, int],
    ):
        self.model = model
        self.n_decimals = n_decimals
        self.intensity_weighting_power = intensity_weighting_power
        self.allowed_missing_percentage = allowed_missing_percentage
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.document_converter = DocumentConverter()
        self.embedding_maker = EmbeddingMaker()

    def predict(self, context, model_input: str):
        embeddings = self._pre_process_data(model_input)
        return embeddings  # temporary
        # get library embeddings from feast
        # compare both embeddings
        # return best_matches for each spectrum

    def _pre_process_data(self, model_input: str):
        loaded_data = self.data_loader.load_gnps_json(model_input)
        cleaned_data = [self.data_cleaner.clean_data(data) for data in loaded_data]
        documents = [
            self.document_converter.convert_to_document(spectrum, self.n_decimals)
            for spectrum in cleaned_data
        ]
        embeddings = [
            self.embedding_maker.make_embedding(
                self.model,
                document,
                self.intensity_weighting_power,
                self.allowed_missing_percentage,
            )
            for document in documents
        ]
        return embeddings
