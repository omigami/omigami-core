from typing import Union, Dict
from urllib.parse import urlparse
from drfs import DRPath
from prefect.engine.result import Result
from prefect.engine.results import S3Result, LocalResult


def create_result(path: str, **kwargs) -> Dict[str, Union[Result, str]]:
    path = DRPath(path)
    protocol = getattr(path, "scheme", "file")

    protocol_to_result = {
        "s3": S3Result,
        "file": LocalResult,
        "redis": None,
    }

    if protocol == "s3":
        # extracts bucket name from s3 path
        directory = path.netloc
        file_name = str(path.relative_to(path.parts[0]))
    elif protocol == "file":
        directory = str(path.parent)
        file_name = path.name
    else:
        raise NotImplementedError(f"Protocol {protocol} is not implemented.")

    return {
        "result": protocol_to_result[protocol](directory, **kwargs),
        "target": file_name,
    }
