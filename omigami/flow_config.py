from datetime import timedelta
from enum import Enum

from attr import dataclass
from drfs import DRPath
from omigami.config import ROOT_DIR, S3_BUCKETS
from prefect.executors import Executor, LocalDaskExecutor
from prefect.run_configs import RunConfig, KubernetesRun
from prefect.schedules import IntervalSchedule
from prefect.storage import Storage, S3

"""
    Implemented Prefect flow configurations:
"""


class PrefectStorageMethods(Enum):
    S3 = 0


class PrefectExecutorMethods(Enum):
    DASK = 0
    LOCAL_DASK = 1


@dataclass
class FlowConfig:
    """
    Configuration options to be passed into Prefect's Flow() as arguments.
    Therefore, should mirror expected Flow() arguments ~exactly~.
    """

    run_config: RunConfig
    storage: Storage
    executor: Executor
    schedule: IntervalSchedule = None

    @property
    def kwargs(self):
        return self.__dict__


def make_flow_config(
    image: str,
    storage_type: PrefectStorageMethods,
    executor_type: PrefectExecutorMethods,
    redis_db: str = "",
    environment: str = "dev",
    schedule: timedelta = None,
) -> FlowConfig:
    """
    This will coordinate the creation of a flow config either with provided params or using the default values
    from default_config.yaml
    """

    # run_config
    run_config = KubernetesRun(
        image=image,
        job_template_path=str(ROOT_DIR / "job_spec.yaml"),
        labels=["dev"],
        service_account_name="prefect-server-serviceaccount",
        env={"REDIS_HOST": "redis-master.redis", "REDIS_DB": redis_db},
        memory_request="12Gi",
    )

    # storage_type
    if storage_type == PrefectStorageMethods.S3:
        storage = S3(DRPath(S3_BUCKETS[environment]).netloc)
    else:
        raise ValueError(f"Prefect flow storage type '{storage_type}' not supported.")

    # executor
    if executor_type == PrefectExecutorMethods.DASK:
        raise NotImplementedError(
            "DASK as a prefect executor is not supported at the moment."
        )
    elif executor_type == PrefectExecutorMethods.LOCAL_DASK:
        executor = LocalDaskExecutor(scheduler="threads", num_workers=5)
    else:
        raise ValueError(f"Prefect flow executor type '{executor_type}' not supported.")

    flow_config = FlowConfig(run_config=run_config, storage=storage, executor=executor)
    if schedule:
        flow_config.schedule = IntervalSchedule(interval=schedule)

    return flow_config
