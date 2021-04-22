from mlflow.pyfunc import PythonModel


class SimpleModel(PythonModel):
    def predict(self, context, model_input: int):
        return model_input + 2

    def set_run_id(self):
        pass
