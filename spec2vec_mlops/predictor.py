import ast
from logging import getLogger
from typing import Union, List, Dict, Any

import numpy as np
from gensim.models import Word2Vec
from matchms import calculate_scores
from matchms.importing.load_from_json import as_spectrum
from matchms.filtering import normalize_intensities
from mlflow.pyfunc import PythonModel

from spec2vec_mlops import config
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

KEYS = config["gnps_json"]["necessary_keys"]

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
        self, data_input_and_parameters: Dict, mz_range: int = 1
    ) -> List[List[Dict]]:
        """Match spectra from a json payload input with spectra having the highest similarity scores
        in the GNPS spectra library.
        Return a list matches of IDs and scores for each input spectrum.
        """
        log.info("Creating a prediction.")
        data_input, parameters = self._parse_input(data_input_and_parameters)
        self._validate_input(data_input)
        log.info("Pre-processing data.")
        input_embeddings = self._pre_process_data(data_input)

        log.info("Loading reference embeddings.")
        ref_spectrum_ids = self._get_ref_ids_from_data_input(data_input, mz_range)
        log.info(f"Loaded {len(ref_spectrum_ids.keys())} IDs from the database.")
        ref_embeddings = self._load_unique_ref_embeddings(ref_spectrum_ids)
        log.info(f"Loaded {len(ref_embeddings.keys())} embeddings from the database.")
        log.info("Getting best matches.")

        all_best_matches = []
        for i, input_emb in enumerate(input_embeddings):
            ref_emb_for_input = self._get_input_ref_embeddings(
                ref_spectrum_ids, ref_embeddings, spectrum_number=i
            )
            input_best_matches = self._get_best_matches(  # SP1+SP2+SP3
                ref_emb_for_input, input_emb, input_spectrum_number=i, **parameters
            )
            all_best_matches.append(input_best_matches)

        log.info("Finishing prediction.")
        return all_best_matches

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    @staticmethod
    def _parse_input(data_input_and_parameters: Dict):
        if not isinstance(data_input_and_parameters, dict):
            data_input_and_parameters = data_input_and_parameters.tolist()

        parameters = (
            data_input_and_parameters.get("parameters")
            if data_input_and_parameters.get("parameters")
            else {}
        )
        data_input = data_input_and_parameters.get("data")
        return data_input, parameters

    def _pre_process_data(self, data_input: List[Dict]) -> List[Embedding]:
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
        self, data_input: List[Dict], mz_range: int = 1
    ) -> Dict[str, List[str]]:
        ref_spectrum_ids_dict = dict()
        for i, spectrum in enumerate(data_input):
            precursor_mz = spectrum["Precursor_MZ"]
            min_mz, max_mz = (
                float(precursor_mz) - mz_range,
                float(precursor_mz) + mz_range,
            )
            ref_ids = self.dgw.get_spectrum_ids_within_range(min_mz, max_mz)
            ref_spectrum_ids_dict[f"refs_spectrum{i}"] = ref_ids
        return ref_spectrum_ids_dict

    def _load_unique_ref_embeddings(
        self, spectrum_ids_dict: Dict[str, List[str]]
    ) -> Dict[str, Embedding]:
        unique_ref_ids = set(
            item for elem in list(spectrum_ids_dict.values()) for item in elem
        )
        log.info(f"Read embeddings of {len(unique_ref_ids)} IDs from the database.")
        unique_ref_embeddings = self.dgw.read_embeddings(
            self.run_id, list(unique_ref_ids)
        )
        log.info(
            f"Creating Dict[id, Embedding] for {len(unique_ref_embeddings)} unique reference embeddings."
        )
        ref_embeddings_dict = dict()
        for emb in unique_ref_embeddings:
            ref_embeddings_dict[emb.spectrum_id] = emb
        return ref_embeddings_dict

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

    @staticmethod
    def _get_input_ref_embeddings(
        ref_spectrum_ids_dict: Dict, ref_embeddings_dict: Dict, spectrum_number: int
    ):
        spectrum_ids = ref_spectrum_ids_dict[f"refs_spectrum{spectrum_number}"]
        log.info(
            f"{len(ref_spectrum_ids_dict[f'refs_spectrum{spectrum_number}'])} "
            f"embeddings for spectrum number {spectrum_number}."
        )
        # log.info(
        #     f"The following spectrum IDs are not present in the reference embeddings: "
        #     f"{set(spectrum_ids) - set(ref_embeddings_dict.keys())}"
        # )
        if spectrum_ids:
            ref_emb_for_input = [
                ref_embeddings_dict[sp_id]
                for sp_id in spectrum_ids
                if sp_id in ref_embeddings_dict
            ]
        else:
            raise RuntimeError(
                f"No data found from filtering with precursor MZ for spectrum number {spectrum_number}. "
                f"Try increasing the mz_range filtering."
            )
        return ref_emb_for_input
