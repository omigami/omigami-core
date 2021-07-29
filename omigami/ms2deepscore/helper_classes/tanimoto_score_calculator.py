from logging import Logger
from typing import List

import numpy as np
import pandas as pd
from ms2deepscore import BinnedSpectrum
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity

from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)


class TanimotoScoreCalculator:
    def __init__(
        self,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        n_bits: int = 2048,
        decimals: int = 5,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._n_bits = n_bits
        self._decimals = decimals

    def calculate(
        self, spectrum_ids: List[str], scores_output_path: str, logger: Logger = None
    ) -> str:
        binned_spectra = self._spectrum_dgw.read_binned_spectra(spectrum_ids)
        unique_inchi_keys = self._get_unique_inchis(binned_spectra)

        if logger:
            logger.info(
                f"Calculating Tanimoto scores for {len(unique_inchi_keys)} unique InChIkeys"
            )

        tanimoto_scores = self._calculate_tanimoto_scores(unique_inchi_keys, logger)
        tanimoto_scores.to_pickle(scores_output_path, compression="gzip")
        return scores_output_path

    @staticmethod
    def _get_unique_inchis(binned_spectra: List[BinnedSpectrum]) -> pd.Series:
        inchi_keys, inchi = zip(
            *[
                (spectrum.get("inchikey")[:14], spectrum.get("inchi"))
                for spectrum in binned_spectra
            ]
        )
        inchi_keys_2_inchi = pd.DataFrame({"inchi": inchi, "inchi_key": inchi_keys})

        most_common_inchi = inchi_keys_2_inchi.groupby(["inchi_key"]).agg(
            pd.Series.mode
        )

        return most_common_inchi["inchi"].apply(
            lambda x: x[0] if isinstance(x, np.ndarray) else x
        )

    def _calculate_tanimoto_scores(
        self,
        inchis: pd.Series,
        logger=None,
    ) -> pd.DataFrame:
        def _derive_daylight_fingerprint(df, nbits: int):
            mol = Chem.MolFromInchi(df)
            try:
                return Chem.RDKFingerprint(mol, fpSize=nbits)
            except:
                logger.info(f"bad inchi = {df}")
                logger.info(f"None mol = = {mol}")

        fingerprints = inchis.apply(
            _derive_daylight_fingerprint,
            args=(self._n_bits,),
        )

        scores = fingerprints.apply(lambda x: BulkTanimotoSimilarity(x, fingerprints))
        scores = pd.DataFrame.from_dict(
            dict(zip(scores.index, scores.values)),
        )
        scores.index = fingerprints.index
        return scores.round(self._decimals)
