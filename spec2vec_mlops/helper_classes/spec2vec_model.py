from typing import List, Dict, Union

from gensim.models import Word2Vec
from matchms import calculate_scores
from mlflow.pyfunc import PythonModel

from spec2vec_mlops.entities.embedding import Embedding
from spec2vec_mlops.helper_classes.data_cleaner import DataCleaner
from spec2vec_mlops.helper_classes.data_loader import DataLoader
from spec2vec_mlops.helper_classes.document_converter import DocumentConverter
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.spec2vec_embeddings import Spec2VecEmbeddings
from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumIDStorer,
    EmbeddingStorer,
)


class Model(PythonModel):
    def __init__(
        self,
        model: Word2Vec,
        n_decimals: int,
        intensity_weighting_power: Union[float, int],
        allowed_missing_percentage: Union[float, int],
        run_id: str = None,
    ):
        self.model = model
        self.n_decimals = n_decimals
        self.intensity_weighting_power = intensity_weighting_power
        self.allowed_missing_percentage = allowed_missing_percentage
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.document_converter = DocumentConverter()
        self.embedding_maker = EmbeddingMaker(self.n_decimals)
        self.run_id = run_id

    def predict(self, context, model_input: List[Dict]) -> List[Dict]:
        embeddings = self._pre_process_data(model_input)
        reference_embeddings = self._get_reference_embeddings()
        best_matches = self._get_best_matches(reference_embeddings, embeddings)
        return best_matches

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    def _pre_process_data(self, model_input: List[Dict]) -> List[Embedding]:
        cleaned_data = [self.data_cleaner.clean_data(data) for data in model_input]
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

    def _get_reference_embeddings(self) -> List[Embedding]:
        spectrum_id_storer = SpectrumIDStorer(
            feature_table_name="spectrum_ids_info",
        )
        embedding_storer = EmbeddingStorer(
            feature_table_name="embedding_info",
            run_id=self.run_id,
        )
        all_spectrum_ids = spectrum_id_storer.read_online()
        embeddings = embedding_storer.read_online(all_spectrum_ids)
        return embeddings

    def _get_best_matches(
        self, references: List[Embedding], queries: List[Embedding], n_best_spectra: int
    ) -> List[List[Dict]]:
        spec2vec_embeddings_similarity = Spec2VecEmbeddings(
            model=self.model,
            intensity_weighting_power=self.intensity_weighting_power,
            allowed_missing_percentage=self.allowed_missing_percentage,
        )
        scores = calculate_scores(
            references,
            queries,
            spec2vec_embeddings_similarity,
        )
        spectra_best_matches = []
        for i, query in enumerate(queries):
            spectrum_best_scores = scores.scores_by_query(query, sort=True)[:n_best_spectra]
            spectrum_best_matches = []
            for spectrum_match in spectrum_best_scores:
                spectrum_best_matches.append(
                    {
                        "spectrum_number": i,
                        "best_match_id": spectrum_match[0].spectrum_id,
                        "score": spectrum_match[1],
                    }
                )
            spectra_best_matches.append(spectrum_best_matches)
        return spectra_best_matches
