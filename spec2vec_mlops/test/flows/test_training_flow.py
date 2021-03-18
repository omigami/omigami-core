from pathlib import Path

import pytest

from spec2vec_mlops.flows.training_flow import spec2vec_train_pipeline_local


@pytest.fixture()
def gnps_small_json():
    ASSET_DIR = str(Path(__file__).parents[1] / "assets" / "SMALL_GNPS.json")
    return f"file://{ASSET_DIR}"


def test_spec2vec_train_pipeline_local(gnps_small_json):
    state = spec2vec_train_pipeline_local(source_uri=gnps_small_json)
    assert state.is_successful()
