from datetime import timedelta

from prefect import Flow
from prefect.schedules import Schedule

from omigami.config import DATASET_IDS
from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
    FlowConfig,
)


def test_make_flow_config():
    flow_config = make_flow_config(
        image="image-ref-name-test-harry-potter-XXII",
        storage_type=PrefectStorageMethods.Local,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db=DATASET_IDS["small"],
        schedule=timedelta(seconds=2),
    )

    assert isinstance(flow_config, FlowConfig)
    assert isinstance(flow_config.schedule, Schedule)
    assert Flow("harry-potter-and-the-forbidden-flow", **flow_config.kwargs)
