from datetime import timedelta

from prefect import Flow
from prefect.schedules import Schedule
from prefect.storage import Local

from omigami.config import DATASET_IDS
from omigami.flow_config import make_flow_config, PrefectExecutorMethods, FlowConfig


def test_make_flow_config():
    flow_config = make_flow_config(
        image="image-ref-name-test-harry-potter-XXII",
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db=DATASET_IDS["small"],
        schedule=timedelta(seconds=2),
    )

    assert isinstance(flow_config, FlowConfig)
    assert isinstance(flow_config.schedule, Schedule)
    assert Flow("harry-potter-and-the-forbidden-flow", **flow_config.kwargs)
    assert isinstance(flow_config.storage, Local)
