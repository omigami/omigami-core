from ms2deepscore.models import SiameseModel

from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)


def test_save_model_local(siamese_model, tmpdir):
    model_path = f"{tmpdir}/model.hdf5"
    fs_gtw = MS2DeepScoreFSDataGateway()
    fs_gtw.save(siamese_model, model_path)

    assert fs_gtw.fs.exists(model_path)


def test_save_model_s3(siamese_model, s3_mock):
    model_path = "s3://test-bucket/model.hdf5"
    fs_gtw = MS2DeepScoreFSDataGateway()
    fs_gtw.save(siamese_model, model_path)

    assert fs_gtw.fs.exists(model_path)


def test_load_model_local(siamese_model_path):
    fs_gtw = MS2DeepScoreFSDataGateway()
    model = fs_gtw.load_model(siamese_model_path)

    assert isinstance(model, SiameseModel)


def test_load_model_s3(siamese_model, s3_mock):
    model_path = "s3://test-bucket/model.hdf5"
    fs_gtw = MS2DeepScoreFSDataGateway()
    fs_gtw.save(siamese_model, model_path)

    model = fs_gtw.load_model(model_path)

    assert isinstance(model, SiameseModel)
