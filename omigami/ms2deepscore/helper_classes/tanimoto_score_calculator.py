import pandas as pd
from typing import List

from ms2deepscore import BinnedSpectrum
from omigami.gateways.data_gateway import SpectrumDataGateway
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity


class TanimotoScoreCalculator:
    def __init__(self, spectrum_dgw: SpectrumDataGateway):
        self._spectrum_dgw = spectrum_dgw

    def calculate(self, n_bits: int = 2048) -> pd.DataFrame:
        binned_spectra = self._spectrum_dgw.read_binned_spectra()
        unique_inchi_keys = self._get_unique_inchis(binned_spectra)
        tanimoto_scores = self._get_tanimoto_scores(unique_inchi_keys, n_bits)
        return tanimoto_scores

    def _get_unique_inchis(self, binned_spectra: List[BinnedSpectrum]) -> pd.Series:
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

    def _get_tanimoto_scores(self, inchis: pd.Series, n_bits: int) -> pd.DataFrame:
        def _derive_daylight_fingerprint(df, nbits: int):
            mol = Chem.MolFromInchi(df)
            return Chem.RDKFingerprint(mol, fpSize=nbits)

        fingerprints = inchis.apply(
            _derive_daylight_fingerprint,
            args=(n_bits,),
        )
        scores = {}
        for i, outer_fingerprint in fingerprints.iteritems():
            scores[i] = {}
            for j, inner_fingerprint in fingerprints.iteritems():
                scores[i][j] = TanimotoSimilarity(outer_fingerprint, inner_fingerprint)
        return pd.DataFrame(scores)
