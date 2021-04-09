import ast
from typing import List, Dict, Union

import flask
from flask import jsonify
from gensim.models import Word2Vec
from matchms import calculate_scores
from mlflow.pyfunc import PythonModel

from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_cleaner import DataCleaner
from spec2vec_mlops.helper_classes.data_loader import DataLoader
from spec2vec_mlops.helper_classes.document_converter import DocumentConverter
from spec2vec_mlops.helper_classes.embedding import Embedding
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.exception import (
    MandatoryKeyMissingError,
    IncorrectInputTypeError,
    IncorrectPeaksJsonTypeError,
    IncorrectFloatFieldTypeError,
    IncorrectStringFieldTypeError,
    ValidateInputException,
)
from spec2vec_mlops.helper_classes.spec2vec_embeddings import Spec2VecEmbeddings

KEYS = config["gnps_json"]["necessary_keys"]


class Model(PythonModel):
    model_error_handler = flask.Blueprint("error_handlers", __name__)

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
        self.embedding_maker = EmbeddingMaker(self.n_decimals)

    def predict(self, context, model_input: List[Dict]) -> List[Dict]:
        self._validate_input(model_input)
        embeddings = self._pre_process_data(model_input)
        # get library embeddings from feast
        # for now going to use the calculated ones
        best_matches = self._get_best_matches(embeddings, embeddings)
        return best_matches

    @staticmethod
    @model_error_handler.app_errorhandler(ValidateInputException)
    def handleCustomError(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

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

    def _get_best_matches(
        self, references: List[Embedding], queries: List[Embedding]
    ) -> List[Dict]:
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
        best_matches = []
        for i, query in enumerate(queries):
            best_match = scores.scores_by_query(query)[0]
            best_matches.append(
                {
                    "spectrum_number": i,
                    "best_match_id": best_match[0].spectrum_id,
                    "score": best_match[1],
                }
            )
        return best_matches

    @staticmethod
    def _validate_input(model_input: List[Dict]):
        for spectrum in model_input:
            if not isinstance(spectrum, Dict):
                raise IncorrectInputTypeError("Input data must be a dictionary", 1, 400)

            mandatory_keys = ["peaks_json", "Precursor_MZ"]
            if any(key not in spectrum.keys() for key in mandatory_keys):
                raise MandatoryKeyMissingError(
                    f"Please include all the mandatory keys in your input data. "
                    f"The mandatory keys are {mandatory_keys}",
                    1,
                    400,
                )

            if isinstance(spectrum["peaks_json"], str):
                try:
                    ast.literal_eval(spectrum["peaks_json"])
                except ValueError:
                    raise IncorrectPeaksJsonTypeError(
                        "peaks_json needs to be a string representation of a list or a list",
                        1,
                        400,
                    )
            elif not isinstance(spectrum["peaks_json"], list):
                raise IncorrectPeaksJsonTypeError(
                    "peaks_json needs to be a string representation of a list or a list",
                    1,
                    400,
                )

            float_keys = ["Precursor_MZ", "Charge"]
            for key in float_keys:
                if spectrum.get(key):
                    try:
                        float(spectrum[key])
                    except ValueError:
                        raise IncorrectFloatFieldTypeError(
                            f"{key} needs to be a string representation of a float",
                            1,
                            400,
                        )

            for key in KEYS:
                if key not in float_keys + mandatory_keys:
                    if not isinstance(spectrum.get(key, ""), str):
                        raise IncorrectStringFieldTypeError(
                            f"{key} needs to be a string", 1, 400
                        )
