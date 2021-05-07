from drfs import DRPath
from prefect.engine.result import Result
from prefect.engine.results import S3Result, LocalResult


def create_result(path: str, **kwargs) -> Result:
    path = DRPath(path)
    protocol = getattr(path, "scheme", "file")

    protocol_to_result = {
        "s3": S3Result,
        "file": LocalResult,
        "redis": None,
    }

    if protocol == "s3":
        # extracts bucket name from s3 path
        directory = str(path).strip("s3://").split("/")[0]
    elif protocol == "file":
        directory = str(path.parent)
    else:
        raise NotImplementedError(f"Protocol {protocol} is not implemented.")

    return protocol_to_result[protocol](directory, **kwargs)
