import os
import random
import string
from copy import deepcopy

import numpy as np
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
    binned_spectra_copy = deepcopy(binned_spectra)
    spectrum = binned_spectra_copy[0]
    for i in range(2):
        new_spectrum = deepcopy(spectrum)
        new_spectrum.set(
            "inchi",
            f"InChI=1S/{''.join(random.choices(string.ascii_uppercase, k=10))}/",
        )
        binned_spectra_copy.append(new_spectrum)

    new_spectrum = deepcopy(spectrum)
    new_spectrum.set("inchi", "InChI=1S/MOSTCOMMON/")
    binned_spectra_copy.extend([new_spectrum] * 3)
    return binned_spectra_copy


@pytest.fixture()
def inchis(binned_spectra):
    inchi_keys, inchis = zip(
        *[
            (spectrum.get("inchikey"), spectrum.get("inchi"))
            for spectrum in binned_spectra
        ]
    )
    return pd.Series(inchis, index=inchi_keys).drop_duplicates()


def test_get_unique_inchi(
    binned_spectra_with_repeated_inchikeys, binned_spectra, tanimoto_calculator
):
    inchis = tanimoto_calculator._get_unique_inchis(
        binned_spectra_with_repeated_inchikeys
    )

    assert "InChI=1S/MOSTCOMMON/" in inchis.values
    assert binned_spectra[0].get("inchi") not in inchis.values


def test_get_tanimoto_scores(inchis, tanimoto_calculator):
    scores = tanimoto_calculator._calculate_tanimoto_scores(inchis, 2048)

    assert isinstance(scores, pd.DataFrame)
    assert (np.diag(scores) == 1).all()
    assert scores.shape[0] == scores.shape[1]
    assert scores.notnull().all().all()


def test_calculate(binned_spectra_stored, tanimoto_calculator, tmpdir):
    path = f"{tmpdir}/tanimoto_scores.pkl"
    tanimoto_calculator.calculate(path)

    assert os.path.exists(path)
