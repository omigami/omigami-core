import os
import pickle
from pathlib import Path

import pytest

from omigami.predictor import Predictor
from omigami.test.conftest import ASSETS_DIR


os.chdir(Path(__file__).parents[3])


@pytest.fixture
def small_payload():
    small_payload = {
        "parameters": {"n_best_spectra": 2},
        "data": [
            {
                "peaks_json": "[[80.060677, 157.0], [337.508301, 230.0]]",
            },
            {
                "peaks_json": "[[81.060677, 157.0], [327.508301, 230.0]]",
            }
        ],
    }
    return small_payload


@pytest.fixture
def big_payload():
    big_payload = {
        "parameters": {"n_best_spectra": 2},
        "data": [
            {
                "peaks_json": "[[80.060677, 157.0], [81.072548, 295.0], [83.088249, 185.0], [91.057564, 601.0], "
                "[95.085892, 634.0], [105.07077, 590.0], [107.09063, 228.0], [109.107414, 238.0], "
                "[115.056114, 482.0], [117.07032, 715.0], [120.779747, 313.0], [121.067802, 257.0], "
                "[121.102318, 430.0], [128.064728, 119.0], [129.074646, 241.0], [130.083542, 291.0], "
                "[259.16922, 175.0], [273.219421, 190.0], [301.217468, 592.0], [1337.508301, 230.0]]",
            },
            {
                "peaks_json": "[[80.060677, 157.0], [81.072548, 295.0], [83.088249, 185.0], [91.057564, 601.0], "
                "[95.085892, 634.0], [105.07077, 590.0], [107.09063, 228.0], [109.107414, 238.0], "
                "[115.056114, 482.0], [117.07032, 715.0], [120.779747, 313.0], [121.067802, 257.0], "
                "[121.102318, 430.0], [128.064728, 119.0], [129.074646, 241.0], [130.083542, 291.0], "
                "[259.16922, 175.0], [273.219421, 190.0], [301.217468, 592.0], [1337.508301, 230.0]]",
            },
        ],
    }
    return big_payload


@pytest.fixture()
def model(word2vec_model):
    return Predictor(
        word2vec_model,
        n_decimals=1,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=25,
        run_id="1",
    )


@pytest.fixture()
def predict_parameters():
    return {"n_best_spectra": 5}


def test_pre_process_data(word2vec_model, model, small_payload):
    data = small_payload["data"].items()
    pre_processed_data = model._pre_process_data(data)
    assert pre_processed_data


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
@pytest.mark.skipif(
    not os.path.exists(str(ASSETS_DIR / "test_model.pkl")),
    reason="test_model.pkl is git ignored",
)
def test_local_predictions(small_payload, big_payload, redis_full_setup):
    path = str(ASSETS_DIR / "test_model.pkl")

    with open(path, "rb") as input_file:
        local_model = pickle.load(input_file)

    local_model.run_id = "1"

    score = local_model.predict(
        data_input_and_parameters=big_payload, mz_range=10, context=""
    )

    assert type(score) == float


def test_parse_input(small_payload, model):
    data_input, parameters = model._parse_input(small_payload)

    assert "peaks_json" in data_input[0]

