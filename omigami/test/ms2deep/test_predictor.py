import os
from pathlib import Path

import pytest
from matchms.importing import load_from_mgf

from ms2deepscore import MS2DeepScore
from omigami.ms2deep.predictor import Predictor
from ms2deepscore.models import load_model
from omigami.test.conftest import ASSETS_DIR


os.chdir(Path(__file__).parents[3])


@pytest.fixture
def spectrums():
    spectrums = list(load_from_mgf(str(ASSETS_DIR / "pesticides_processed.mgf")))
    return spectrums


@pytest.fixture
def payload(spectrums):
    spectrum_a = spectrums[0]
    spectrum_b = spectrums[1]
    payload = {
        "parameters": {},
        "data": [
            {
                "intensities": spectrum_a.peaks.intensities,
                "mz": spectrum_a.peaks.mz,
            },
            {
                "intensities": spectrum_b.peaks.intensities,
                "mz": spectrum_b.peaks.mz,
            },
        ],
    }
    return payload


@pytest.fixture()
def model():
    path = str(ASSETS_DIR / "ms2deepscore_model.hdf5")
    model = load_model(path)
    model = MS2DeepScore(model)
    return Predictor(model=model)


@pytest.mark.skipif(
    not os.path.exists(str(ASSETS_DIR / "ms2deepscore_model.hdf5")),
    reason="ms2deepscore_model.hdf5 is git ignored",
)
def test_predictions(model, payload):
    model.run_id = "1"

    score = model.predict(
        data_input_and_parameters=payload,
        context="",
    )

    assert type(score) == float


def test_parse_input(payload, model):
    data_input, parameters = model._parse_input(payload)

    assert len(data_input) == 2
    assert "intensities" in data_input[0]
    assert "mz" in data_input[0]
