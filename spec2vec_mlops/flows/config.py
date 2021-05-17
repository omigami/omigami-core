from enum import Enum

from attr import dataclass
from prefect.executors import Executor, DaskExecutor, LocalDaskExecutor
from prefect.run_configs import RunConfig, KubernetesRun
from prefect.storage import Storage, S3

from spec2vec_mlops import config, ROOT_DIR

S3_MODEL_BUCKET = config["prefect"]["s3_model_bucket"]

"""
    Implemented Prefect flow configurations:
"""


class PrefectRunMethods(Enum):
    KUBERNETES = 0


class PrefectStorageMethods(Enum):
    S3 = 0


class PrefectExecutorMethods(Enum):
    DASK = 0
    LOCAL_DASK = 1


@dataclass
class FlowConfig:
    """
    Configuration options to be passed into Prefect's Flow() as arguments.

    Therefore, should mirror expected Flow() arguments ~exatcly~.
    """

    run_config: RunConfig
    storage: Storage
    executor: Executor


def make_flow_config(
    image: str,
    run_config_type: PrefectRunMethods,
    storage_type: PrefectStorageMethods,
    executor_type: PrefectExecutorMethods,
    redis_db: str,
) -> FlowConfig:
    """
    This will coordinate the creation of a flow config either with provided params or using the default values
    from default_config.yaml
    """

    # run_config
    if run_config_type == PrefectRunMethods.KUBERNETES:
        run_config = KubernetesRun(
            image=image,
            job_template_path=str(ROOT_DIR / "job_spec.yaml"),
            labels=["dev"],
            service_account_name="prefect-server-serviceaccount",
            env={"REDIS_HOST": "redis-master.redis", "REDIS_DB": redis_db},
        )
    else:
        raise ValueError(
            f"Prefect flow run config type '{run_config_type}' not supported."
        )

    # storage_type
    if storage_type == PrefectStorageMethods.S3:
        storage = S3(S3_MODEL_BUCKET)
    else:
        raise ValueError(f"Prefect flow storage type '{storage_type}' not supported.")

    # executor
    if executor_type == PrefectExecutorMethods.DASK:
        executor = DaskExecutor(address="dask-scheduler.dask:8786")
    elif executor_type == PrefectExecutorMethods.LOCAL_DASK:
        executor = LocalDaskExecutor(scheduler="threads", num_workers=5)
    else:
        raise ValueError(f"Prefect flow executor type '{executor_type}' not supported.")

    return FlowConfig(run_config=run_config, storage=storage, executor=executor)
