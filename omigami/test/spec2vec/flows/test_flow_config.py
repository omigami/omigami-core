from datetime import timedelta

from prefect import Flow
from prefect.schedules import Schedule

from omigami.spec2vec.config import DATASET_IDS
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
        redis_db=DATASET_IDS["dev"]["small"],
        schedule=timedelta(seconds=2),
    )

    assert isinstance(flow_config, FlowConfig)
    assert isinstance(flow_config.schedule, Schedule)
    assert Flow("harry-potter-and-the-forbidden-flow", **flow_config.kwargs)
