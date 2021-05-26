from logging import getLogger
from typing import Union, List, Dict, Any, Tuple

import numpy as np
from gensim.models import Word2Vec
from matchms import calculate_scores
from matchms.importing.load_from_json import as_spectrum
from matchms.filtering import normalize_intensities
from mlflow.pyfunc import PythonModel

from spec2vec_mlops.entities.embedding import Embedding
from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.spec2vec_embeddings import Spec2VecEmbeddings


log = getLogger(__name__)


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
        self.dgw = RedisSpectrumDataGateway()

    def predict(
        self,
        context,
        data_input_and_parameters: Dict[str, Union[Dict, List]],
        mz_range: int = 1,
    ) -> List[List[Dict]]:
        """Match spectra from a json payload input with spectra having the highest similarity scores
        in the GNPS spectra library.
        Return a list matches of IDs and scores for each input spectrum.
        """
        log.info("Creating a prediction.")
        data_input, parameters = self._parse_input(data_input_and_parameters)
        log.info("Pre-processing data.")
        input_spectra_embeddings = self._pre_process_data(data_input)

        log.info("Loading reference embeddings.")
        reference_spectra_ids = self._get_ref_ids_from_data_input(data_input, mz_range)
        self.check_spectrum_refs(reference_spectra_ids)
        log.info(f"Loaded {len(reference_spectra_ids)} IDs from the database.")
        reference_embeddings = self._load_unique_ref_embeddings(reference_spectra_ids)
        log.info(
            f"Loaded {len(reference_embeddings.keys())} embeddings from the database."
        )
        log.info("Getting best matches.")

        all_best_matches = []
        for ref_spectrum_ids, input_spectrum_emb in zip(
            reference_spectra_ids, input_spectra_embeddings
        ):
            input_spectrum_ref_emb = self._get_input_ref_embeddings(
                ref_spectrum_ids, reference_embeddings
            )
            input_spectrum_number = reference_spectra_ids.index(ref_spectrum_ids) + 1
            spectrum_best_matches = self._get_best_matches(  # SP1+SP2+SP3
                input_spectrum_ref_emb,
                input_spectrum_emb,
                input_spectrum_number,
                **parameters,
            )
            all_best_matches.append(spectrum_best_matches)

        log.info("Finishing prediction.")
        return all_best_matches

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    @staticmethod
    def _parse_input(
        data_input_and_parameters: Dict[str, Union[Dict, List]]
    ) -> Tuple[Union[dict, list, None], Union[dict, list, None, dict]]:
        if not isinstance(data_input_and_parameters, dict):
            data_input_and_parameters = data_input_and_parameters.tolist()

        parameters = (
            data_input_and_parameters.get("parameters")
            if data_input_and_parameters.get("parameters")
            else {}
        )
        data_input = data_input_and_parameters.get("data")
        return data_input, parameters

    def _pre_process_data(self, data_input: List[Dict[str, str]]) -> List[Embedding]:
        cleaned_data = [as_spectrum(data) for data in data_input]
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

    def _get_ref_ids_from_data_input(
        self, data_input: List[Dict[str, str]], mz_range: int = 1
    ) -> List[List[str]]:
        ref_spectrum_ids = []
        for i, spectrum in enumerate(data_input):
            precursor_mz = spectrum["Precursor_MZ"]
            min_mz, max_mz = (
                float(precursor_mz) - mz_range,
                float(precursor_mz) + mz_range,
            )
            ref_ids = self.dgw.get_spectrum_ids_within_range(min_mz, max_mz)
            ref_spectrum_ids.append(ref_ids)
        return ref_spectrum_ids

    @staticmethod
    def check_spectrum_refs(reference_spectra_ids: List[List[str]]):
        if [] in reference_spectra_ids:
            idx_null = [
                idx
                for idx, element in enumerate(reference_spectra_ids)
                if element == []
            ]
            raise RuntimeError(
                f"No data found from filtering with precursor MZ for spectra at indices {idx_null}. "
                f"Try increasing the mz_range filtering."
            )

    def _load_unique_ref_embeddings(
        self, spectrum_ids: List[List[str]]
    ) -> Dict[str, Embedding]:
        unique_ref_ids = set(item for elem in spectrum_ids for item in elem)
        unique_ref_embeddings = self.dgw.read_embeddings(
            self.run_id, list(unique_ref_ids)
        )
        return {emb.spectrum_id: emb for emb in unique_ref_embeddings}

    def _get_best_matches(
        self,
        references: List[Embedding],
        query: Embedding,
        input_spectrum_number: int,
        n_best_spectra: int = 10,
        **parameters,
    ) -> List[Dict[str, Union[int, Any]]]:
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
        spectrum_best_matches = []
        for spectrum_match in spectrum_best_scores:
            spectrum_best_matches.append(
                {
                    "spectrum_input": input_spectrum_number,
                    "match_spectrum_id": spectrum_match[0].spectrum_id,
                    "score": spectrum_match[1],
                }
            )
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
