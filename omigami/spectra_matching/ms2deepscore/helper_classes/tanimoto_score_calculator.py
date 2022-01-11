from logging import Logger
from typing import List

import pandas as pd
from ms2deepscore import BinnedSpectrum
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity

from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import MS2DeepScoreFSDataGateway


class TanimotoScoreCalculator:
    def __init__(
        self,
        fs_dgw: MS2DeepScoreFSDataGateway,
        binned_spectra_path: str,
        n_bits: int = 2048,
        decimals: int = 5,
    ):
        self._fs_dgw = fs_dgw
        self._binned_spectra_path = binned_spectra_path
        self._n_bits = n_bits
        self._decimals = decimals

    def calculate(
        self, spectrum_ids: List[str], scores_output_path: str, logger: Logger = None
    ) -> str:
        binned_spectra = self._fs_dgw.read_from_file(self._binned_spectra_path)
        unique_inchi_keys = self._get_unique_inchis(binned_spectra)

        if logger:
            logger.info(
                f"Calculating Tanimoto scores for {len(unique_inchi_keys)} unique InChIkeys"
            )

        tanimoto_scores = self._calculate_tanimoto_scores(unique_inchi_keys)
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

        def custom_mode(x):
            mode = x.mode()
            return mode[0] if len(mode) > 1 else mode

        most_common_inchi = inchi_keys_2_inchi.groupby(["inchi_key"]).agg(custom_mode)

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
