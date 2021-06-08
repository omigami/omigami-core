from datetime import timedelta

from prefect import Flow
from prefect.schedules import IntervalSchedule

from omigami.flows.config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
    FlowConfig,
)


def test_make_flow_config():
    flow_config = make_flow_config(
        image="image-ref-name-test-harry-potter-XXII",
        storage_type=PrefectStorageMethods.S3,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db="2",
        schedule=IntervalSchedule(interval=timedelta(seconds=2)),
    )

    dict_flow_config = flow_config.__dict__

    assert isinstance(flow_config, FlowConfig)
    assert Flow("harry-potter-and-the-forbidden-flow", **dict_flow_config)
