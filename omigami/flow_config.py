from datetime import timedelta
from enum import Enum

from attr import dataclass
from prefect.executors import Executor, LocalDaskExecutor
from prefect.run_configs import RunConfig, KubernetesRun, LocalRun, DockerRun
from prefect.schedules import IntervalSchedule
from prefect.storage import Storage, S3, Local, Docker

from omigami.config import (
    ROOT_DIR,
    STORAGE_ROOT,
    OMIGAMI_ENV,
    MLFLOW_SERVER,
    REDIS_HOST,
)
from omigami.env_config import Environments

"""
    Implemented Prefect flow configurations:
"""


class PrefectStorageMethods(Enum):
    S3 = 0
    Local = 1
    Docker = 2


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
    executor_type: PrefectExecutorMethods,
    image: str = None,
    redis_db: str = "",
    storage_root: str = None,
    schedule: timedelta = None,
) -> FlowConfig:
    """
    Creates the configuration necessary to run an Omigami prefect flow.

    Parameters
    ----------
    executor_type:
        Dask executor. Only local is implemented
    image:
        Image to run the flow on. Used by k8s flows. Unused by local flows
    redis_db:
        Redis database where the data is
    storage_root:
        Root directory of flow persistence
    schedule:
        Optional parameter for running a flow periodically

    Returns
    -------
    FlowConfig:
        A class containing flow configuration parameters

    """

    env_variables = {
        "REDIS_DB": redis_db,
        "REDIS_HOST": REDIS_HOST,
        "OMIGAMI_ENV": OMIGAMI_ENV,
        "MLFLOW_SERVER": MLFLOW_SERVER,
        "TZ": "UTC",
    }

    if OMIGAMI_ENV in (Environments.dev, Environments.prod):
        storage = S3(STORAGE_ROOT.netloc)
        run_config = KubernetesRun(
            image=image,
            job_template_path=str(ROOT_DIR / "job_spec.yaml"),
            labels=["dev"],
            service_account_name="prefect-server-serviceaccount",
            env=env_variables,
            memory_request="12Gi",
            image_pull_secrets=["regcred"],
        )
    elif OMIGAMI_ENV == Environments.local:
        storage = Local(storage_root)
        run_config = LocalRun(env=env_variables, labels=["dev"])
    elif OMIGAMI_ENV == Environments.docker:
        storage = Docker(
            base_image=image,
            local_image=True,
            ignore_healthchecks=False,
            prefect_directory="/opt/omigami",
        )
        run_config = DockerRun(env=env_variables, labels=["dev"], image=image)
    else:
        raise ValueError(f"Environment {OMIGAMI_ENV} not supported.")

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
