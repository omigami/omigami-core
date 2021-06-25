from omigami.ms2deepscore.config import MS2DEEPSCORE_MODEL_URI
from prefect import Flow

from omigami.data_gateway import InputDataGateway
from omigami.flow_config import FlowConfig
from omigami.ms2deepscore.tasks import (
    DownloadPreTrainedModel,
    DownloadPreTrainedModelParameters,
    DeployModel,
    DeployModelParameters,
    RegisterModel,
)


class MinimalFlowParameters:
    def __init__(
        self,
        input_dgw: InputDataGateway,
        model_output_dir: str,
        model_uri: str = MS2DEEPSCORE_MODEL_URI,
        overwrite: bool = False,
        environment: str = "dev",
    ):
        self.input_dgw = input_dgw

        self.downloading = DownloadPreTrainedModelParameters(
            model_uri,
            model_output_dir,
        )
        self.deploying = DeployModelParameters(overwrite, environment)


def build_minimal_flow(
    project_name: str,
    flow_name: str,
    flow_config: FlowConfig,
    flow_parameters: MinimalFlowParameters,
    model_output_dir: str,
    mlflow_server: str,
    deploy_model: bool = False,
) -> Flow:
    """
    Builds a minimal MS2DeepScore machine learning pipeline. It downloads a pre
    trained model, registers it and deploys it to the API.


    Parameters
    ----------
    project_name: str
        Prefect parameter. The project name.
    flow_name:
        Name of the flow
    flow_config: FlowConfig
        Configuration dataclass passed to prefect.Flow as a dict
    flow_parameters:
        Class containing all flow parameters
    model_output_dir:
        Directory for saving the model.
    mlflow_server:
        Server used for MLFlow to save the model.
    deploy_model:
        Whether to create a seldon deployment with the result of the training flow.
    Returns
    -------

    """
    with Flow(flow_name, **flow_config.kwargs) as training_flow:
        ms2deepscore_model_path = DownloadPreTrainedModel(
            flow_parameters.input_dgw,
            flow_parameters.downloading,
        )()

        model_registry = RegisterModel(
            project_name,
            model_output_dir,
            mlflow_server,
            flow_parameters.downloading.output_path,
        )(ms2deepscore_model_path)

        if deploy_model:
            DeployModel(flow_parameters.deploying)(model_registry)

    return training_flow
