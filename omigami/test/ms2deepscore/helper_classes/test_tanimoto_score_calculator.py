import string
import random
from copy import deepcopy

import pandas as pd
import pytest
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.helper_classes.tanimoto_score_calculator import (
    TanimotoScoreCalculator,
)


@pytest.fixture()
def tanimoto_calculator():
    return TanimotoScoreCalculator(spectrum_dgw=MS2DeepScoreRedisSpectrumDataGateway())


@pytest.fixture()
def binned_spectra_with_repeated_inchikeys(binned_spectra):
    spectrum = binned_spectra[0]
    for i in range(2):
        new_spectrum = deepcopy(spectrum)
        new_spectrum.set(
            "inchi",
            f"InChI=1S/{''.join(random.choices(string.ascii_uppercase, k=10))}/",
        )
        binned_spectra.append(new_spectrum)

    new_spectrum = deepcopy(spectrum)
    new_spectrum.set("inchi", "InChI=1S/MOSTCOMMON/")
    binned_spectra.extend([new_spectrum] * 3)
    return binned_spectra


@pytest.fixture()
def inchis(binned_spectra):
    return set([spectrum.get("inchi") for spectrum in binned_spectra])


def test_get_unique_inchi(
    binned_spectra_with_repeated_inchikeys, binned_spectra, tanimoto_calculator
):
    inchis = tanimoto_calculator._get_unique_inchis(
        binned_spectra_with_repeated_inchikeys
    )

    assert "InChI=1S/MOSTCOMMON/" in inchis
    assert binned_spectra[0].get("inchi") not in inchis


def test_get_tanimoto_scores(inchis, tanimoto_calculator):
    scores = tanimoto_calculator._get_tanimoto_scores(inchis)

    assert isinstance(scores, pd.DataFrame)


def test_calculate(binned_spectra_stored):
    pass
