from typing import List, Dict

from gensim.models import Word2Vec
from matchms import calculate_scores
from mlflow.pyfunc import PythonModel

from spec2vec_mlops.helper_classes.data_cleaner import DataCleaner
from spec2vec_mlops.helper_classes.document_converter import DocumentConverter
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker


class Model(PythonModel):
    def __init__(self, model: Word2Vec, n_decimals: int):
        self.model = model
        self.n_decimals = n_decimals
        self.data_cleaner = DataCleaner()
        self.document_converter = DocumentConverter()
        self.embedding_maker = EmbeddingMaker()

    def predict(self, context, model_input: List[Dict]):
        cleaned_data = [self.data_cleaner.clean_data(data) for data in model_input]
        documents = [self.document_converter.convert_to_document(spectrum, self.n_decimals) for spectrum in cleaned_data]
        embeddings = [self.embedding_maker.make_embedding(self.model, document, 0.5, 5.0) for document in documents]
        # get library embeddings from feast
        # compare both embeddings
        # return best_matches for each spectrum
        #scores = calculate_scores(spectrum_documents, spectrum_documents, spec2vec_similarity, is_symmetric=True)
        #best_matches = scores.scores_by_query(spectrum_documents[11], sort=True)[:10]
