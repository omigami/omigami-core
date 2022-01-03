from pydantic import BaseModel


"""
This is an idea of where/how we can store parameters definitions. This wont need to be
used until we start working on jobsystem and/or frontend

```
parameters = PyMUVRParameters(n_iter=n_iter)
parameters.json("s3://export_path.json")
```

"""


class PyMUVRParameter(BaseModel):
    name: str
    description: str


class NumberOfIterations(PyMUVRParameter):
    value: int
    name = "n_iter"
    description = "Number of Iterations"


class PyMUVRFlowParameters(BaseModel):
    def __init__(self, n_iter: int):
        self.n_iter = NumberOfIterations(n_iter)
