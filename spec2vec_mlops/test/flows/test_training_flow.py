from spec2vec_mlops import config
from spec2vec_mlops.flows.training_flow import spec2vec_train_pipeline_local

FEAST_CORE_URL_LOCAL = config["feast"]["url"]["local"].get(str)


def test_spec2vec_train_pipeline_local(gnps_small_json, tmpdir):
    state = spec2vec_train_pipeline_local(
        source_uri=gnps_small_json,
        feast_source_dir=f"file://{tmpdir}",
        feast_core_url=FEAST_CORE_URL_LOCAL,
    )
    assert state.is_successful()
