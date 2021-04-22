import mlflow
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PythonModel


class SimpleModel(PythonModel):
    def predict(self, context, model_input: int):
        return model_input + 2


class ModelRegister:
    def __init__(self, server_uri: str):
        mlflow.set_tracking_uri(server_uri)

    def register_model(
        self,
        model: SimpleModel,
        experiment_name: str,
        path: str,
    ) -> str:
        experiment_id = self._get_or_create_experiment_id(experiment_name, path)
        with mlflow.start_run(experiment_id=experiment_id) as run:
            params = {
                "parameter1": 100,
                "parameter2": 5,
                "parameter3": 1,
            }
            mlflow.log_params(params)
            run_id = run.info.run_id
            try:
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=model,
                    registered_model_name=experiment_name,
                    code_path=[
                        "mlops_presentation",
                    ],
                )
                # This is need to run the flow locally. mlflow.pyfunc.log_model is not supported without a database
            except MlflowException:
                mlflow.pyfunc.save_model(
                    f"{path}/model",
                    python_model=model,
                    code_path=[
                        "mlops_presentation",
                    ],
                )
            metrics = {
                "metric1": 1,
                "metric2": 80,
                "metric3": 6,
            }
            mlflow.log_metrics(metrics)
            return run_id

    @staticmethod
    def _get_or_create_experiment_id(experiment_name: str, path: str) -> str:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name, artifact_location=path)
