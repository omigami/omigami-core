from logging import getLogger
from typing import Union, List, Dict, Tuple

import flask
import numpy as np
from flask import jsonify
from gensim.models import Word2Vec
from matchms import calculate_scores
from matchms.filtering import normalize_intensities
from matchms.importing.load_from_json import as_spectrum

from omigami.spectra_matching.predictor import (
    Predictor,
    SpectrumMatches,
    SpectraMatchingPredictorException,
)
from omigami.spectra_matching.spec2vec import SPEC2VEC_PROJECT_NAME
from omigami.spectra_matching.spec2vec.entities.embedding import Spec2VecEmbedding
from omigami.spectra_matching.spec2vec.entities.spectrum_document import (
    SpectrumDocumentData,
)
from omigami.spectra_matching.spec2vec.helper_classes.embedding_maker import (
    EmbeddingMaker,
    EmbeddingMakerError,
)
from omigami.spectra_matching.spec2vec.helper_classes.similarity_score_calculator import (
    Spec2VecSimilarityScoreCalculator,
)
from omigami.spectra_matching.storage import RedisSpectrumDataGateway

log = getLogger(__name__)


class Spec2VecPredictor(Predictor):
    def __init__(
        self,
        model: Word2Vec,
        ion_mode: str,
        n_decimals: int,
        intensity_weighting_power: Union[float, int],
        allowed_missing_percentage: Union[float, int],
        run_id: str = None,
    ):
        self.model = model
        self.ion_mode = ion_mode
        self.n_decimals = n_decimals
        self.intensity_weighting_power = intensity_weighting_power
        self.allowed_missing_percentage = allowed_missing_percentage
        self.embedding_maker = EmbeddingMaker(self.n_decimals)
        self._run_id = run_id
        super().__init__(RedisSpectrumDataGateway(SPEC2VEC_PROJECT_NAME))

        model_error_handler = flask.Blueprint("error_handlers", __name__)

        @model_error_handler.app_errorhandler(SpectraMatchingPredictorException)
        def handleCustomError(error):
            log.info("I am handling the error.")
            response = jsonify(error.to_dict())
            response.status_code = error.status_code
            return response

    def predict(
        self,
        context,
        data_input_and_parameters: Dict[str, Union[Dict, List]],
        mz_range: int = 1,
    ) -> Dict[str, SpectrumMatches]:
        """Match spectra from a json payload input with spectra having the highest
        similarity scores in the GNPS spectra library. Return a list matches of IDs
        and scores for each input spectrum.
        """
        if mz_range == 1:
            log.warning("I'm going to error.")
            raise SpectraMatchingPredictorException("I am a bad error", 1, 404)
        try:
            log.info("Creating a prediction.")
            data_input, parameters = self._parse_input(data_input_and_parameters)
            log.info("Pre-processing data.")
            input_spectra_embeddings = self._pre_process_data(data_input)

            log.info("Loading reference embeddings.")
            reference_spectra_ids = self._get_ref_ids_from_data_input(
                data_input, mz_range
            )
            log.info(f"Loaded {len(reference_spectra_ids)} IDs from the database.")
            reference_embeddings = self._load_unique_ref_embeddings(
                reference_spectra_ids
            )
            log.info(
                f"Loaded {len(reference_embeddings)} embeddings from the database."
            )

            log.info("Calculating best matches.")
            best_matches = {}

            for i, input_spectrum in enumerate(input_spectra_embeddings):

                input_spectrum_ref_emb = self._get_input_ref_embeddings(
                    reference_spectra_ids[i], reference_embeddings
                )

                best_matches_data = {
                    "references": input_spectrum_ref_emb,
                    "query": input_spectrum,
                }

                if parameters.get("n_best_spectra"):
                    best_matches_data["n_best_spectra"] = parameters.get(
                        "n_best_spectra"
                    )

                spectrum_best_matches = self._calculate_best_matches(
                    **best_matches_data
                )

                best_matches[
                    input_spectrum.spectrum_id or f"spectrum-{i}"
                ] = spectrum_best_matches

            best_matches = self._add_metadata(best_matches)

            log.info("Finishing prediction.")
            return best_matches
        except (RuntimeError, FileNotFoundError, ValueError, EmbeddingMakerError) as e:
            # TODO carefully pick http status code per exception type
            raise SpectraMatchingPredictorException(str(e), 1, 404)
        except Exception as e:
            # TODO carefully pick http status code per exception type
            raise SpectraMatchingPredictorException(str(e), 2, 500)

    @staticmethod
    def _parse_input(
        data_input_and_parameters: Dict[str, Union[Dict, List]]
    ) -> Tuple[Union[dict, list, None], Union[dict, list, None, dict]]:
        if isinstance(data_input_and_parameters, np.ndarray):
            data_input_and_parameters = data_input_and_parameters.tolist()
        parameters = data_input_and_parameters.get("parameters", {})
        data_input = data_input_and_parameters.get("data")
        return data_input, parameters

    def _pre_process_data(
        self, data_input: List[Dict[str, str]]
    ) -> List[Spec2VecEmbedding]:
        embeddings = []
        for data in data_input:
            raw_spectrum = as_spectrum(data)
            if raw_spectrum:
                norm_spectrum = normalize_intensities(raw_spectrum)
                spectrum_data = SpectrumDocumentData(norm_spectrum, self.n_decimals)
                embeddings.append(
                    self.embedding_maker.make_embedding(
                        self.model,
                        spectrum_data.document,
                        self.intensity_weighting_power,
                        self.allowed_missing_percentage,
                    )
                )
        return embeddings

    def _load_unique_ref_embeddings(
        self, spectrum_ids: List[List[str]]
    ) -> Dict[str, Spec2VecEmbedding]:
        unique_ref_ids = set(item for elem in spectrum_ids for item in elem)
        unique_ref_embeddings = self.dgw.read_embeddings(
            self.ion_mode, list(unique_ref_ids)
        )
        return {emb.spectrum_id: emb for emb in unique_ref_embeddings}

    def _calculate_best_matches(
        self,
        references: List[Spec2VecEmbedding],
        query: Spec2VecEmbedding,
        n_best_spectra: int = 10,
    ) -> SpectrumMatches:
        spec2vec_similarity_score_calculator = Spec2VecSimilarityScoreCalculator(
            model=self.model,
            intensity_weighting_power=self.intensity_weighting_power,
            allowed_missing_percentage=self.allowed_missing_percentage,
        )
        scores = calculate_scores(
            references,
            [query],
            spec2vec_similarity_score_calculator,
        )

        all_scores = scores.scores_by_query(query, sort=True)
        all_scores = [(em, sc) for em, sc in all_scores if not np.isnan(sc)]
        spectrum_best_scores = all_scores[:n_best_spectra]
        spectrum_best_matches = {}
        for spectrum_match in spectrum_best_scores:
            spectrum_best_matches[spectrum_match[0].spectrum_id] = {
                "score": spectrum_match[1]
            }
        return spectrum_best_matches

    @staticmethod
    def _get_input_ref_embeddings(
        ref_spectrum_ids: List[str],
        ref_embeddings: Dict[str, Spec2VecEmbedding],
    ) -> List[Spec2VecEmbedding]:
        ref_emb_for_input = [
            ref_embeddings[sp_id]
            for sp_id in ref_spectrum_ids
            if sp_id in ref_embeddings
        ]
        return ref_emb_for_input
