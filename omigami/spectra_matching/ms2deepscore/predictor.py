from logging import getLogger
from typing import Union, List, Dict, Tuple

import numpy as np
from matchms import calculate_scores
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model as ms2deepscore_load_model, SiameseModel
from tqdm import tqdm

from omigami.spectra_matching.ms2deepscore.embedding import (
    MS2DeepScoreEmbedding,
    EmbeddingMaker,
)
from omigami.spectra_matching.ms2deepscore.helper_classes.spectrum_processor import (
    SpectrumProcessor,
)
from omigami.spectra_matching.ms2deepscore.storage import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.predictor import Predictor, SpectrumMatches
from omigami.spectra_matching.util import (
    cosine_similarity,
    cosine_similarity_matrix,
)

log = getLogger(__name__)


class MS2DeepScorePredictor(Predictor):
    def __init__(self, ion_mode: str = None, run_id: str = None):
        super().__init__(MS2DeepScoreRedisSpectrumDataGateway())
        self.ion_mode = ion_mode
        self._run_id = run_id
        self.spectrum_processor = SpectrumProcessor()
        self.embedding_maker = EmbeddingMaker()
        self.model: Union[SiameseModel, None] = None

    def load_context(self, context):
        model_path = context.artifacts["ms2deepscore_model_path"]
        try:
            log.info(f"Loading model from {model_path}")
            self.model = ms2deepscore_load_model(model_path)
        except FileNotFoundError:
            log.error(f"Could not find MS2DeepScore model in {model_path}")

    def predict(
        self,
        context,
        data_input: Dict[str, List],
        mz_range: int = 1,
    ) -> Dict:
        """Match spectra from a json payload input with spectra having the highest
        structural similarity scores in the GNPS spectra library. Return a list
        matches of IDs and scores for each input spectrum. An example of data_input
        is as follows:
        {
            "data": [
                {
                    "peaks_json": List[float],
                    "Precursor_MZ": float,
                },
                {
                    "peaks_json": List[float],
                    "Precursor_MZ": float,
                },
            ],
            "parameters": {"n_best_spectra": int, "include_metadata": List[str]}
        }

        Returns
        -------
        ms2ds_similarity
            MS2DeepScore similarity score.
        """
        log.info("Creating a prediction.")
        data_input, parameters = self._parse_input(data_input)

        log.info("Loading reference spectra.")
        reference_spectra_ids = self._get_ref_ids_from_data_input(data_input, mz_range)
        reference_embeddings = self._load_embeddings(reference_spectra_ids)
        log.info(f"Loaded {len(reference_embeddings)} spectra from the database.")

        log.info("Pre-processing data.")
        query_spectra = self.spectrum_processor.process_spectra(
            data_input, process_reference_spectra=False
        )
        query_binned_spectra = self.model.spectrum_binner.transform(query_spectra)
        query_embeddings = [
            self.embedding_maker.make_embedding(self.model, binned_spectrum)
            for binned_spectrum in query_binned_spectra
        ]

        log.info("Calculating best matches.")

        best_matches_data = {
            "all_references": reference_embeddings,
            "queries": query_embeddings,
        }

        if parameters.get("n_best_spectra"):
            best_matches_data["n_best_spectra"] = parameters.get("n_best_spectra")

        best_matches = self._calculate_best_matches(**best_matches_data)
        best_matches = self._add_metadata(best_matches)

        log.info("Finishing prediction.")
        return best_matches

    @staticmethod
    def _parse_input(
        data_input_and_parameters: Dict[str, Union[Dict, List]]
    ) -> Tuple[Union[dict, list, None], Union[dict, list, None, dict]]:
        data_input = data_input_and_parameters.get("data")
        parameters = data_input_and_parameters.get("parameters")

        return data_input, parameters

    def _calculate_best_matches(
        self,
        all_references: List[MS2DeepScoreEmbedding],
        queries: List[MS2DeepScoreEmbedding],
        n_best_spectra: int = 10,
    ) -> SpectrumMatches:
        similarity_score_calculator = MS2DeepScoreSimilarityScoreCalculator(self.model)

        scores = calculate_scores(
            all_references,
            queries,
            similarity_score_calculator,
        )
        best_matches = {}
        for i, query in enumerate(queries):
            all_scores = scores.scores_by_query(query, sort=True)

            all_scores = [
                (spectrum, score)
                for spectrum, score in all_scores
                if not np.isnan(score)
            ]
            spectrum_best_scores = all_scores[:n_best_spectra]
            spectrum_best_matches = {}
            for spectrum_match in spectrum_best_scores:
                spectrum_best_matches[spectrum_match[0].spectrum_id] = {
                    "score": spectrum_match[1]
                }
            best_matches[f"spectrum-{i}"] = spectrum_best_matches
        return best_matches

    def _load_embeddings(
        self, spectrum_ids: List[List[str]]
    ) -> List[MS2DeepScoreEmbedding]:
        unique_ids = set(item for elem in spectrum_ids for item in elem)
        embeddings = self.dgw.read_embeddings(self.ion_mode, list(unique_ids))
        return embeddings


class MS2DeepScoreSimilarityScoreCalculator(MS2DeepScore):
    """Calculate MS2DeepScore similarity scores between a reference and a query. The
    only difference between MS2DeepScoreSimilarityScoreCalculator and MS2DeepScore is that
    MS2DeepScoreSimilarityScoreCalculator methods take as input argument Embedding instead
    of Spectrum.
    """

    def __init__(self, model: SiameseModel, **kwargs):
        super().__init__(model, **kwargs)

    def pair(
        self, reference: MS2DeepScoreEmbedding, query: MS2DeepScoreEmbedding
    ) -> float:
        return cosine_similarity(reference.vector[0, :], query.vector[0, :])

    def matrix(
        self,
        references: List[MS2DeepScoreEmbedding],
        queries: List[MS2DeepScoreEmbedding],
        is_symmetric: bool = False,
    ) -> np.ndarray:

        reference_vectors = self.calculate_vectors(references)
        if is_symmetric:
            assert np.all(
                references == queries
            ), "Expected references to be equal to queries for is_symmetric=True"
            query_vectors = reference_vectors
        else:
            query_vectors = self.calculate_vectors(queries)

        ms2ds_similarity = cosine_similarity_matrix(reference_vectors, query_vectors)
        return ms2ds_similarity

    def calculate_vectors(
        self, spectrum_list: List[MS2DeepScoreEmbedding]
    ) -> np.ndarray:
        n_rows = len(spectrum_list)
        reference_vectors = np.empty((n_rows, self.output_vector_dim), dtype="float")
        for index_reference, reference in enumerate(
            tqdm(
                spectrum_list,
                desc="Calculating vectors of reference spectrums",
                disable=(not self.progress_bar),
            )
        ):
            reference_vectors[
                index_reference, 0 : self.output_vector_dim
            ] = reference.vector
        return reference_vectors
