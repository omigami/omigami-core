from logging import getLogger
from typing import Union, List, Dict, Tuple

import numpy as np
from gensim.models import Word2Vec
from matchms import calculate_scores
from matchms.filtering import normalize_intensities
from matchms.importing.load_from_json import as_spectrum

from omigami.gateways import RedisSpectrumDataGateway
from omigami.predictor import Predictor, SpectrumMatches
from omigami.spec2vec.entities.embedding import Embedding
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData


from omigami.spec2vec.helper_classes.embedding_maker import EmbeddingMaker
from omigami.spec2vec.helper_classes.spec2vec_embeddings import Spec2VecEmbeddings
from omigami.spec2vec.config import PROJECT_NAME

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
        super().__init__(RedisSpectrumDataGateway(PROJECT_NAME))

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
        log.info("Creating a prediction.")
        data_input, parameters = self._parse_input(data_input_and_parameters)
        log.info("Pre-processing data.")
        input_spectra_embeddings = self._pre_process_data(data_input)

        log.info("Loading reference embeddings.")
        reference_spectra_ids = self._get_ref_ids_from_data_input(data_input, mz_range)
        log.info(f"Loaded {len(reference_spectra_ids)} IDs from the database.")
        reference_embeddings = self._load_unique_ref_embeddings(reference_spectra_ids)
        log.info(f"Loaded {len(reference_embeddings)} embeddings from the database.")

        log.info("Calculating best matches.")
        best_matches = {}

        for i, input_spectrum in enumerate(input_spectra_embeddings):

            input_spectrum_ref_emb = self._get_input_ref_embeddings(
                reference_spectra_ids[i], reference_embeddings
            )
            spectrum_best_matches = self._calculate_best_matches(
                references=input_spectrum_ref_emb,
                query=input_spectrum,
            )
            best_matches[
                input_spectrum.spectrum_id or f"spectrum-{i}"
            ] = spectrum_best_matches

        if parameters.get("include_metadata", None):
            best_matches = self._add_metadata(
                best_matches, parameters["include_metadata"]
            )

        log.info("Finishing prediction.")
        return best_matches

    def set_run_id(self, run_id: str):
        self._run_id = run_id

    @staticmethod
    def _parse_input(
        data_input_and_parameters: Dict[str, Union[Dict, List]]
    ) -> Tuple[Union[dict, list, None], Union[dict, list, None, dict]]:
        if not isinstance(data_input_and_parameters, dict):
            data_input_and_parameters = data_input_and_parameters.tolist()

        parameters = data_input_and_parameters.get("parameters", {})
        data_input = data_input_and_parameters.get("data")
        return data_input, parameters

    def _pre_process_data(self, data_input: List[Dict[str, str]]) -> List[Embedding]:
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
    ) -> Dict[str, Embedding]:
        unique_ref_ids = set(item for elem in spectrum_ids for item in elem)
        unique_ref_embeddings = self.dgw.read_embeddings(
            self.ion_mode, self._run_id, list(unique_ref_ids)
        )
        return {emb.spectrum_id: emb for emb in unique_ref_embeddings}

    def _calculate_best_matches(
        self,
        references: List[Embedding],
        query: Embedding,
        n_best_spectra: int = 10,
    ) -> SpectrumMatches:
        spec2vec_embeddings_similarity = Spec2VecEmbeddings(
            model=self.model,
            intensity_weighting_power=self.intensity_weighting_power,
            allowed_missing_percentage=self.allowed_missing_percentage,
        )
        scores = calculate_scores(
            references,
            [query],
            spec2vec_embeddings_similarity,
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
        ref_embeddings: Dict[str, Embedding],
    ) -> List[Embedding]:
        ref_emb_for_input = [
            ref_embeddings[sp_id]
            for sp_id in ref_spectrum_ids
            if sp_id in ref_embeddings
        ]
        return ref_emb_for_input
