from enum import Enum

from attr import dataclass
from drfs import DRPath
from prefect.executors import Executor, LocalDaskExecutor
from prefect.run_configs import RunConfig, KubernetesRun
from prefect.storage import Storage, S3

from omigami.config import ROOT_DIR, S3_BUCKET


"""
    Implemented Prefect flow configurations:
"""


class PrefectStorageMethods(Enum):
    S3 = 0


class PrefectExecutorMethods(Enum):
    DASK = 0
    LOCAL_DASK = 1


class IonModes:
    positive = "positive"
    negative = "negative"


@dataclass
class FlowConfig:
    """
    Configuration options to be passed into Prefect's Flow() as arguments.

    Therefore, should mirror expected Flow() arguments ~exactly~.
    """

    run_config: RunConfig
    storage: Storage
    executor: Executor

    @property
    def kwargs(self):
        return self.__dict__


def make_flow_config(
    image: str,
    storage_type: PrefectStorageMethods,
    executor_type: PrefectExecutorMethods,
    redis_db: str,
    environment: str = "dev",
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
    )

    # storage_type
    if storage_type == PrefectStorageMethods.S3:
        storage = S3(DRPath(S3_BUCKET[environment]).netloc)
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

    return FlowConfig(run_config=run_config, storage=storage, executor=executor)
