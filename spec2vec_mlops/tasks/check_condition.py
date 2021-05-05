from typing import List, Any, Optional

from prefect import task


@task()
def check_condition(inputs: Optional[List[Any]]):
    return True if inputs and len(inputs) > 0 else False
