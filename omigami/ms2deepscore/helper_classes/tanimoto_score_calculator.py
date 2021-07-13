import pandas as pd
from typing import List

from matchms.utils import derive_fingerprint_from_inchi
from ms2deepscore import BinnedSpectrum
from omigami.gateways.data_gateway import SpectrumDataGateway
from rdkit.DataStructs import TanimotoSimilarity


class TanimotoScoreCalculator:
    def __init__(self, spectrum_dgw: SpectrumDataGateway):
        self._spectrum_dgw = spectrum_dgw

    def calculate(self) -> pd.DataFrame:
        binned_spectra = self._spectrum_dgw.read_binned_spectra()
        unique_inchi_keys = self._get_unique_inchis(binned_spectra)
        tanimoto_scores = self._get_tanimoto_scores(unique_inchi_keys)
        return tanimoto_scores

    def _get_unique_inchis(self, binned_spectra: List[BinnedSpectrum]) -> List[str]:
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

        return most_common_inchi["inchi"].values.tolist()

    def _get_tanimoto_scores(self, inchis: List[str]) -> pd.DataFrame:
        fingerprints = [
            derive_fingerprint_from_inchi(
                inchi, fingerprint_type="daylight", nbits=2048
            )
            for inchi in inchis
        ]

        # calculate score for all pairs
        return pd.DataFrame()
