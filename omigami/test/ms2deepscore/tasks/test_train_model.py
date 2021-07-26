import os
from unittest.mock import Mock

import pytest
from drfs.filesystems import get_fs
from prefect import Flow

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
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
        model_path, spectrum_binner_output_path=fitted_spectrum_binner_path, epochs=2
    )

    with Flow("test") as flow:
        TrainModel(
            fs_gtw=FSDataGateway(),
            spectrum_dgw=MS2DeepScoreRedisSpectrumDataGateway(),
            train_parameters=parameters,
        )([], tanimoto_scores_path)

    state = flow.run()
    assert state.is_successful()
    assert os.path.exists(model_path)


def test_save_model_local(ms2deepscore_model, tmpdir):
    model_path = f"{tmpdir}/model.hdf5"
    parameters = TrainModelParameters(model_path, spectrum_binner_output_path="")
    task = TrainModel(Mock(), Mock(), parameters)
    task._save_model(ms2deepscore_model.model)

    fs = get_fs(model_path)
    assert fs.exist(model_path)


def test_save_model_s3(ms2deepscore_model, s3_mock):
    model_path = "s3://test-bucket/model.hdf5"
    parameters = TrainModelParameters(model_path, spectrum_binner_output_path="")
    task = TrainModel(Mock(), Mock(), parameters)
    task._save_model(ms2deepscore_model.model)

    fs = get_fs(model_path)
    assert fs.exist(model_path)
