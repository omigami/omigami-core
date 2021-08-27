import os

import pytest
from prefect import Flow

from omigami.ms2deepscore.gateways.fs_data_gateway import MS2DeepScoreFSDataGateway
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.ms2deepscore.helper_classes.siamese_model_trainer import (
    SplitRatio,
)
from omigami.ms2deepscore.tasks.train_model import TrainModel, TrainModelParameters


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_train_model(
    tmpdir,
    binned_spectra_to_train_stored,
    tanimoto_scores_path,
    fitted_spectrum_binner_path,
):
    model_path = f"{tmpdir}/model.hdf5"
    parameters = TrainModelParameters(
        model_path,
        "positive",
        spectrum_binner_output_path=fitted_spectrum_binner_path,
        epochs=2,
        split_ratio=SplitRatio(0.6, 0.3, 0.1),
    )

    with Flow("test") as flow:
        TrainModel(
            fs_gtw=MS2DeepScoreFSDataGateway(),
            spectrum_dgw=MS2DeepScoreRedisSpectrumDataGateway(),
            train_parameters=parameters,
        )([], tanimoto_scores_path)

    state = flow.run()
    assert state.is_successful()
    assert os.path.exists(model_path)
