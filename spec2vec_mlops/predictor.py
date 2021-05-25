import ast
import gc
from logging import getLogger
from typing import Union, List, Dict

import numpy as np
from gensim.models import Word2Vec
from matchms import calculate_scores
from matchms.importing.load_from_json import as_spectrum
from matchms.filtering import normalize_intensities
from mlflow.pyfunc import PythonModel

from spec2vec_mlops.config import NECESSARY_KEYS
from spec2vec_mlops.entities.embedding import Embedding
from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.exception import (
    MandatoryKeyMissingException,
    IncorrectPeaksJsonTypeException,
    IncorrectFloatFieldTypeException,
    IncorrectStringFieldTypeException,
    IncorrectSpectrumDataTypeException,
)
from spec2vec_mlops.helper_classes.spec2vec_embeddings import Spec2VecEmbeddings


log = getLogger(__name__)

# REFERENCE_EMBEDDINGS: Optional[List[Embedding]] = None


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
        self.embedding_maker = EmbeddingMaker(self.n_decimals)
        self.run_id = run_id

    def predict(self, context, model_input_and_parameters: Dict) -> List[List[Dict]]:
        log.info("Creating a prediction.")
        if not isinstance(model_input_and_parameters, dict):
            model_input_and_parameters = model_input_and_parameters.tolist()

        parameters = (
            model_input_and_parameters.get("parameters")
            if model_input_and_parameters.get("parameters")
            else {}
        )
        model_input = model_input_and_parameters.get("data")
        self._validate_input(model_input)
        log.info("Pre-processing data.")
        embeddings = self._pre_process_data(model_input)
        log.info("Loading reference embeddings.")
        # global REFERENCE_EMBEDDINGS
        # if REFERENCE_EMBEDDINGS is None:
        reference_embeddings = []
        for spectra in model_input:
            precursor_mz = spectra["Precursor_MZ"]
            reference_embeddings += self._get_reference_embeddings(precursor_mz)
        log.info(f"Loaded {len(reference_embeddings)} from the database.")
        log.info("Getting best matches.")
        best_matches = self._get_best_matches(
            reference_embeddings, embeddings, **parameters
        )
        log.info("Finishing prediction.")
        del embeddings
        gc.collect()
        return best_matches

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    def _pre_process_data(self, model_input: List[Dict]) -> List[Embedding]:
        cleaned_data = [as_spectrum(data) for data in model_input]
        cleaned_data = [normalize_intensities(data) for data in cleaned_data if data]
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

    def _get_reference_embeddings(self, precursor_mz: str) -> List[Embedding]:
        dgw = RedisSpectrumDataGateway()
        min_mz, max_mz = float(precursor_mz) - 1, float(precursor_mz) + 1
        embeddings_iter = dgw.read_embeddings_within_range(self.run_id, min_mz, max_mz)
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
            all_scores = [(em, sc) for em, sc in all_scores if not np.isnan(sc)]
            spectrum_best_scores = all_scores[:n_best_spectra]
            spectrum_best_matches = []
            for spectrum_match in spectrum_best_scores:
                spectrum_best_matches.append(
                    {
                        "spectrum_input": i,
                        "match_spectrum_id": spectrum_match[0].spectrum_id,
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

            for key in NECESSARY_KEYS:
                if key not in float_keys + mandatory_keys:
                    if not isinstance(spectrum.get(key, ""), str):
                        raise IncorrectStringFieldTypeException(
                            f"{key} needs to be a string", 400
                        )
