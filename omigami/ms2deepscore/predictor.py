from logging import getLogger
from typing import Union, List, Dict, Tuple

import numpy as np
from matchms import calculate_scores
from ms2deepscore.models import load_model as ms2deepscore_load_model
from omigami.ms2deepscore.entities.embedding import Embedding
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.helper_classes.embedding_maker import EmbeddingMaker
from omigami.ms2deepscore.helper_classes.ms2deepscore_embedding import (
    MS2DeepScoreEmbedding,
)
from omigami.ms2deepscore.helper_classes.spectrum_processor import (
    SpectrumProcessor,
)
from omigami.predictor import Predictor, SpectrumMatches

log = getLogger(__name__)


class MS2DeepScorePredictor(Predictor):
    def __init__(self, ion_mode: str = None, run_id: str = None):
        super().__init__(MS2DeepScoreRedisSpectrumDataGateway())
        self.ion_mode = ion_mode
        self.run_id = run_id
        self.spectrum_processor = SpectrumProcessor()
        self.embedding_maker = EmbeddingMaker()
        self.model = None

    def load_context(self, context):
        model_path = context.artifacts["ms2deepscore_model_path"]
        try:
            log.info(f"Loading model from {model_path}")
            siamese_model = ms2deepscore_load_model(model_path)
            self.model = MS2DeepScoreEmbedding(siamese_model)
        except FileNotFoundError:
            log.error(f"Could not find MS2DeepScore model in {model_path}")

    def set_run_id(self, run_id: str):
        self.run_id = run_id

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
            "parameters": {"n_best": int, "include_metadata": List[str]}
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
        query_binned_spectra = self.model.model.spectrum_binner.transform(query_spectra)
        query_embeddings = [
            self.embedding_maker.make_embedding(self.model, binned_spectrum)
            for binned_spectrum in query_binned_spectra
        ]

        log.info("Calculating best matches.")
        best_matches = self._calculate_best_matches(
            reference_embeddings,
            query_embeddings,
        )

        if parameters.get("include_metadata", None):
            best_matches = self._add_metadata(
                best_matches, parameters["include_metadata"]
            )

        log.info("Finishing prediction.")
        return best_matches

    @staticmethod
    def _parse_input(
        data_input_and_parameters: Dict[str, Union[Dict, List]]
    ) -> Tuple[Union[dict, list, None], Union[dict, list, None, dict]]:
        if not isinstance(data_input_and_parameters, dict):
            data_input_and_parameters = data_input_and_parameters.tolist()

        data_input = data_input_and_parameters.get("data")
        parameters = data_input_and_parameters.get("parameters")

        return data_input, parameters

    def _calculate_best_matches(
        self,
        all_references: List[Embedding],
        queries: List[Embedding],
        n_best_spectra: int = 10,
    ) -> SpectrumMatches:

        scores = calculate_scores(
            all_references,
            queries,
            self.model,
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

    def _load_embeddings(self, spectrum_ids: List[List[str]]) -> List[Embedding]:
        unique_ids = set(item for elem in spectrum_ids for item in elem)
        embeddings = self.dgw.read_embeddings(
            self.ion_mode, self.run_id, list(unique_ids)
        )
        return embeddings
