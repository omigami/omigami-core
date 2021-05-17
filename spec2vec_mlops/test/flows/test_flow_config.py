from prefect import Flow

from spec2vec_mlops.flows.config import (
    make_flow_config,
    PrefectRunMethods,
    PrefectStorageMethods,
    PrefectExecutorMethods,
    FlowConfig,
)


def test_make_flow_config():
    flow_config = make_flow_config(
        image="image-ref-name-test-harry-potter-XXII",
        run_config_type=PrefectRunMethods.KUBERNETES,
        storage_type=PrefectStorageMethods.S3,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db="2",
    )

    dict_flow_config = flow_config.__dict__

    assert isinstance(flow_config, FlowConfig)
    assert Flow("harry-potter-and-the-forbidden-flow", **dict_flow_config)
