import ast
from typing import Union, List, Dict

from gensim.models import Word2Vec
from matchms import calculate_scores
from mlflow.pyfunc import PythonModel

from spec2vec_mlops import config
from spec2vec_mlops.entities.embedding import Embedding
from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.gateways.redis_gateway import RedisSpectrumDataGateway
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.exception import (
    MandatoryKeyMissingException,
    IncorrectPeaksJsonTypeException,
    IncorrectFloatFieldTypeException,
    IncorrectStringFieldTypeException,
    IncorrectSpectrumDataTypeException,
)
from spec2vec_mlops.helper_classes.spec2vec_embeddings import Spec2VecEmbeddings
from spec2vec_mlops.tasks.process_spectrum.spectrum_processor import SpectrumProcessor

KEYS = config["gnps_json"]["necessary_keys"]


class Predictor(PythonModel):
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
        self.data_cleaner = SpectrumProcessor()
        self.embedding_maker = EmbeddingMaker(self.n_decimals)
        self.run_id = run_id

    def predict(self, context, model_input_and_parameters: Dict) -> List[List[Dict]]:
        parameters = model_input_and_parameters.get("parameters")
        model_input = model_input_and_parameters.get("data")
        self._validate_input(model_input)
        embeddings = self._pre_process_data(model_input)
        reference_embeddings = self._get_reference_embeddings()
        best_matches = self._get_best_matches(
            reference_embeddings, embeddings, **parameters
        )
        return best_matches

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    def _pre_process_data(self, model_input: List[Dict]) -> List[Embedding]:
        cleaned_data = [self.data_cleaner.process_data(data) for data in model_input]
        spectra_data = [
            SpectrumDocumentData(spectrum, self.n_decimals) for spectrum in cleaned_data
        ]
        embeddings = [
            self.embedding_maker.make_embedding(
                self.model,
                spectrum_data.document,
                self.intensity_weighting_power,
                self.allowed_missing_percentage,
            )
            for spectrum_data in spectra_data
        ]
        return embeddings

    def _get_reference_embeddings(self) -> List[Embedding]:
        dgw = RedisSpectrumDataGateway()
        embeddings_iter = dgw.read_embeddings(self.run_id)
        return list(embeddings_iter)

    def _get_best_matches(
        self,
        references: List[Embedding],
        queries: List[Embedding],
        n_best_spectra: int = 10,
        **parameters,
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
            all_scores = scores.scores_by_query(query, sort=True)
            spectrum_best_scores = all_scores[:n_best_spectra]
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

    @staticmethod
    def _validate_input(model_input: List[Dict]):
        for i, spectrum in enumerate(model_input):
            if not isinstance(spectrum, Dict):
                raise IncorrectSpectrumDataTypeException(
                    f"Spectrum data must be a dictionary", 400
                )

            mandatory_keys = ["peaks_json", "Precursor_MZ"]
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
