from ms2deepscore.models import SiameseModel

from omigami.spectra_matching.ms2deepscore.gateways.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)


def test_save_model_local(ms2deepscore_embedding, tmpdir):
    model_path = f"{tmpdir}/model.hdf5"
    fs_gtw = MS2DeepScoreFSDataGateway()
    fs_gtw.save(ms2deepscore_embedding.model, model_path)

    assert fs_gtw.fs.exists(model_path)


def test_save_model_s3(ms2deepscore_embedding, s3_mock):
    model_path = "s3://test-bucket/model.hdf5"
    fs_gtw = MS2DeepScoreFSDataGateway()
    fs_gtw.save(ms2deepscore_embedding.model, model_path)

    assert fs_gtw.fs.exists(model_path)


def test_load_model_local(ms2deepscore_model_path):
    fs_gtw = MS2DeepScoreFSDataGateway()
    model = fs_gtw.load_model(ms2deepscore_model_path)

    assert isinstance(model, SiameseModel)


def test_load_model_s3(ms2deepscore_embedding, s3_mock):
    model_path = "s3://test-bucket/model.hdf5"
    fs_gtw = MS2DeepScoreFSDataGateway()
    fs_gtw.save(ms2deepscore_embedding.model, model_path)

    model = fs_gtw.load_model(model_path)

    assert isinstance(model, SiameseModel)
