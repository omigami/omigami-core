from typing import List

import pandas as pd
from ms2deepscore import BinnedSpectrum
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity


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
        self,
        spectrum_ids: List[str],
        scores_output_path: str,
    ) -> str:
        binned_spectra = self._spectrum_dgw.read_binned_spectra(spectrum_ids)
        unique_inchi_keys = self._get_unique_inchis(binned_spectra)
        tanimoto_scores = self._calculate_tanimoto_scores(unique_inchi_keys)
        tanimoto_scores.to_pickle(scores_output_path)
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

        return most_common_inchi["inchi"]

    def _calculate_tanimoto_scores(
        self,
        inchis: pd.Series,
    ) -> pd.DataFrame:
        def _derive_daylight_fingerprint(df, nbits: int):
            mol = Chem.MolFromInchi(df)
            return Chem.RDKFingerprint(mol, fpSize=nbits)

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
