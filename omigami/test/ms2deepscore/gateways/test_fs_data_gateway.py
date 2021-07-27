from omigami.ms2deepscore.gateways.fs_data_gateway import MS2DeepScoreFSDataGateway


def test_save_model_local(ms2deepscore_model, tmpdir):
    model_path = f"{tmpdir}/model.hdf5"
    fs_gtw = MS2DeepScoreFSDataGateway()
    fs_gtw.save_model(ms2deepscore_model.model, model_path)

    assert fs_gtw.fs.exists(model_path)


def test_save_model_s3(ms2deepscore_model, s3_mock):
    model_path = "s3://test-bucket/model.hdf5"
    fs_gtw = MS2DeepScoreFSDataGateway()
    fs_gtw.save_model(ms2deepscore_model.model, model_path)

    assert fs_gtw.fs.exists(model_path)
