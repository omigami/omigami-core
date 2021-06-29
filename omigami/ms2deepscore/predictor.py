from logging import getLogger
from typing import Union, List, Dict, Tuple

import numpy as np
from matchms import calculate_scores
from matchms.Spectrum import Spectrum
from ms2deepscore.models import load_model as ms2deepscore_load_model
from ms2deepscore import MS2DeepScore
from omigami.gateways.redis_spectrum_data_gateway import RedisSpectrumDataGateway
from omigami.predictor import Predictor, SpectrumMatches

log = getLogger(__name__)


class MS2DeepScorePredictor(Predictor):
    def __init__(self, run_id: str = None):
        super().__init__(RedisSpectrumDataGateway())
        self.run_id = run_id
        self.model = None

    def load_context(self, context):
        model_path = context.artifacts["ms2deepscore_model_path"]
        try:
            log.info(f"Loading model from {model_path}")
            siamese_model = ms2deepscore_load_model(model_path)
            self.model = MS2DeepScore(siamese_model)
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
                    "intensities": List[float],
                    "mz": List[float],
                    "Precursor_MZ": float,
                },
                {
                    "intensities": List[float],
                    "mz": List[float],
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
        log.info(f"Loaded {len(reference_spectra_ids)} IDs from the database.")

        reference_spectra = self._load_unique_spectra(reference_spectra_ids)
        log.info(f"Loaded {len(reference_spectra)} spectra from the database.")

        log.info("Pre-processing data.")
        query_spectra = self._clean_spectra(data_input)
        # TODO: clean reference spectra

        log.info("Calculating best matches.")
        best_matches = {}
        for i, input_spectrum in enumerate(query_spectra):
            spectrum_best_matches = self._calculate_best_matches(
                list(reference_spectra.values()),
                input_spectrum,
            )
            best_matches[f"spectrum-{i}"] = spectrum_best_matches

        if parameters.get("include_metadata", None):
            best_matches = self._add_metadata(
                best_matches, parameters["include_metadata"]
            )

        log.info("Finishing prediction.")
        return best_matches

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    @staticmethod
    def _parse_input(
        data_input_and_parameters: Dict[str, Union[Dict, List]]
    ) -> Tuple[Union[dict, list, None], Union[dict, list, None, dict]]:
        if not isinstance(data_input_and_parameters, dict):
            data_input_and_parameters = data_input_and_parameters.tolist()

        data_input = data_input_and_parameters.get("data")
        parameters = data_input_and_parameters.get("parameters")

        return data_input, parameters

    def _clean_spectra(self, data_input: List[Dict]) -> List[Spectrum]:
        return [
            Spectrum(
                np.asarray(spectrum["mz"]),
                np.asarray(spectrum["intensities"]),
            )
            for spectrum in data_input
        ]

    def _calculate_best_matches(
        self,
        references: List[Spectrum],
        query: Spectrum,
        n_best_spectra: int = 10,
    ) -> SpectrumMatches:

        scores = calculate_scores(
            references,
            [query],
            self.model,
        )
        all_scores = scores.scores_by_query(query, sort=True)
        all_scores = [(em, sc) for em, sc in all_scores if not np.isnan(sc)]
        spectrum_best_scores = all_scores[:n_best_spectra]
        spectrum_best_matches = {}
        for spectrum_match in spectrum_best_scores:
            spectrum_best_matches[spectrum_match[0].metadata["spectrum_id"]] = {
                "score": spectrum_match[1]
            }
        return spectrum_best_matches

    def _load_unique_spectra(self, spectrum_ids: List[List[str]]):
        unique_ids = set(item for elem in spectrum_ids for item in elem)
        return self.dgw.read_spectra(list(unique_ids))
