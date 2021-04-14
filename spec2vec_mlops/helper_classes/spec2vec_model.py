import ast
from typing import Union, List, Dict, Optional

from gensim.models import Word2Vec
from matchms import calculate_scores
from mlflow.pyfunc import PythonModel

from spec2vec_mlops.entities.embedding import Embedding
from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_cleaner import DataCleaner
from spec2vec_mlops.helper_classes.data_loader import DataLoader
from spec2vec_mlops.helper_classes.document_converter import DocumentConverter
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.exception import (
    MandatoryKeyMissingException,
    IncorrectPeaksJsonTypeException,
    IncorrectFloatFieldTypeException,
    IncorrectStringFieldTypeException,
    IncorrectSpectrumDataTypeException,
    IncorrectSpectrumNameTypeException,
    IncorrectDataLengthException,
)
from spec2vec_mlops.helper_classes.spec2vec_embeddings import Spec2VecEmbeddings
from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumIDStorer,
    EmbeddingStorer,
)

KEYS = config["gnps_json"]["necessary_keys"]


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

    def predict(
        self, context: Optional[List[str]], model_input: List[Dict]
    ) -> List[Dict]:
        self._validate_input(context, model_input)
        embeddings = self._pre_process_data(model_input)
        reference_embeddings = self._get_reference_embeddings()
        best_matches = self._get_best_matches(reference_embeddings, embeddings, context)
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
        all_spectrum_ids = spectrum_id_storer.read()
        embeddings = embedding_storer.read(all_spectrum_ids)
        return embeddings

    def _get_best_matches(
        self, references: List[Embedding], queries: List[Embedding], context: List[str]
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
            best_match = scores.scores_by_query(query, sort=True)[0]
            best_matches.append(
                {
                    "spectrum_name": context[i] if isinstance(context, list) else i,
                    "best_match_id": best_match[0].spectrum_id,
                    "score": best_match[1],
                }
            )
        return best_matches

    @staticmethod
    def _validate_input(context: Optional[List[str]], model_input: List[Dict]):
        if isinstance(context, list):
            if len(context) != len(model_input):
                raise IncorrectDataLengthException(
                    "Names and ndarray must have the same length", 400
                )

        for i, spectrum in enumerate(model_input):
            if isinstance(context, list):
                if not isinstance(context[i], str):
                    raise IncorrectSpectrumNameTypeException(
                        "Spectrum name must be a string", 400
                    )

            if not isinstance(spectrum, Dict):
                raise IncorrectSpectrumDataTypeException(
                    "Spectrum data must be a dictionary", 400
                )

            mandatory_keys = ["peaks_json"]
            if any(key not in spectrum.keys() for key in mandatory_keys):
                raise MandatoryKeyMissingException(
                    f"Please include all the mandatory keys in your input data. "
                    f"The mandatory keys are {mandatory_keys}",
                    400,
                )

            if isinstance(spectrum["peaks_json"], str):
                try:
                    ast.literal_eval(spectrum["peaks_json"])
                except ValueError:
                    raise IncorrectPeaksJsonTypeException(
                        "peaks_json needs to be a string representation of a list or a list",
                        400,
                    )
            elif not isinstance(spectrum["peaks_json"], list):
                raise IncorrectPeaksJsonTypeException(
                    "peaks_json needs to be a string representation of a list or a list",
                    400,
                )

            float_keys = ["Precursor_MZ", "Charge"]
            for key in float_keys:
                if spectrum.get(key):
                    try:
                        float(spectrum[key])
                    except ValueError:
                        raise IncorrectFloatFieldTypeException(
                            f"{key} needs to be a string representation of a float",
                            400,
                        )

            for key in KEYS:
                if key not in float_keys + mandatory_keys:
                    if not isinstance(spectrum.get(key, ""), str):
                        raise IncorrectStringFieldTypeException(
                            f"{key} needs to be a string", 400
                        )
