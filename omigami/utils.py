from typing import Dict, Union

from drfs import DRPath
from prefect.engine.result import Result
from prefect.engine.results import S3Result, LocalResult, PrefectResult

from omigami.config import DEFAULT_PREFECT_TASK_CONFIG, OMIGAMI_ENV


def add_click_options(options):
    """
    Helper function to get a list of click.options and add then to cli
    """

    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def create_prefect_result_from_path(
    path: str, **kwargs
) -> Dict[str, Union[Result, str]]:
    """
    Create a Prefect Result object from path.
    Supported Results: S3Result and LocalResult
    """

    path = DRPath(path)
    protocol = getattr(path, "scheme", "file")
    if OMIGAMI_ENV == "docker":
        protocol = "docker"

    protocol_to_result = {
        "s3": S3Result,
        "file": LocalResult,
        "docker": PrefectResult,
    }

    if protocol == "s3":
        # extracts bucket name from s3 path
        args = [path.netloc]
        file_name = str(path.relative_to(path.parts[0]))
    elif protocol == "file":
        args = [str(path.parent)]
        file_name = path.name
    elif protocol == "docker":
        args = []
        file_name = path.name

    else:
        raise NotImplementedError(f"Protocol {protocol} is not implemented.")

    return {
        "result": protocol_to_result[protocol](*args, **kwargs),
        "target": file_name,
    }


def merge_prefect_task_configs(additional_config: dict) -> dict:
    """helper function to merge DEFAULT CONFIG with additional task configurations"""
    return {**DEFAULT_PREFECT_TASK_CONFIG.copy(), **additional_config}
