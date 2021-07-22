import os

import pytest
from prefect import Flow

from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.tasks.train_model import TrainModel, TrainModelParameters


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_train_model(
    tmpdir, binned_spectra_to_train_stored, tanimoto_scores_path, fitted_spectrum_binner
):
    model_path = f"{tmpdir}/model.hdf5"
    print(model_path)
    parameters = TrainModelParameters(model_path, epochs=2)

    with Flow("test") as flow:
        TrainModel(
            spectrum_dgw=MS2DeepScoreRedisSpectrumDataGateway(),
            train_parameters=parameters,
        )([], tanimoto_scores_path, fitted_spectrum_binner)

    state = flow.run()
    assert state.is_successful()
    assert os.path.exists(model_path)
