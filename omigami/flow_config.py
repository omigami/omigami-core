from datetime import timedelta
from enum import Enum

from attr import dataclass
from prefect.executors import Executor, LocalDaskExecutor
from prefect.run_configs import RunConfig, KubernetesRun, LocalRun
from prefect.schedules import IntervalSchedule
from prefect.storage import Storage, S3, Local
from typing_extensions import Literal

from omigami.config import ROOT_DIR, STORAGE_ROOT, ENV

"""
    Implemented Prefect flow configurations:
"""


class PrefectStorageMethods(Enum):
    S3 = 0
    Local = 1


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
    storage_type: PrefectStorageMethods,
    executor_type: PrefectExecutorMethods,
    image: str = None,
    redis_db: str = "",
    schedule: timedelta = None,
    run_env: Literal["k8s", "local"] = "local",
) -> FlowConfig:
    """
    This will coordinate the creation of a flow config either with provided params or using the default values
    from default_config.yaml
    """

    # run_config
    env_variables = {
        "REDIS_HOST": "redis-master.redis",
        "REDIS_DB": redis_db,
        "OMIGAMI_ENV": ENV
    }
    if run_env == "k8s":
        run_config = KubernetesRun(
            image=image,
            job_template_path=str(ROOT_DIR / "job_spec.yaml"),
            labels=["dev"],
            service_account_name="prefect-server-serviceaccount",
            env=env_variables,
            memory_request="12Gi",
        )
    elif run_env == "local":
        run_config = LocalRun(env=env_variables)

    # storage_type
    if storage_type == PrefectStorageMethods.S3:
        storage = S3(STORAGE_ROOT.netloc)
    elif storage_type == PrefectStorageMethods.Local:
        storage = Local(STORAGE_ROOT)
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
