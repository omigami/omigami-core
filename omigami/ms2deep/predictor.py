from logging import getLogger
from typing import Any

from mlflow.pyfunc import PythonModel

log = getLogger(__name__)


class Predictor(PythonModel):
    def __init__(self):
        pass

    def predict(self, context, **kwargs) -> Any:
        """TODO"""
        pass
