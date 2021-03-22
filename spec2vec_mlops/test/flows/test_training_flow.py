from spec2vec_mlops.flows.training_flow import spec2vec_train_pipeline_local


def test_spec2vec_train_pipeline_local(gnps_small_json):
    state = spec2vec_train_pipeline_local(source_uri=gnps_small_json)
    assert state.is_successful()
