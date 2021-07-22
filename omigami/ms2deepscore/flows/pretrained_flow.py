from prefect import Flow

from omigami.flow_config import FlowConfig
from omigami.gateways.data_gateway import InputDataGateway
from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.ms2deepscore.tasks import (
    DeployModel,
    DeployModelParameters,
    RegisterModel,
    ProcessSpectrumParameters,
    ProcessSpectrum,
)


class PretrainedFlowParameters:
    def __init__(
        self,
        input_dgw: InputDataGateway,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        model_uri: str,
        overwrite_model: bool = False,
        environment: str = "dev",
        overwrite_all_spectra: bool = False,
        redis_db: str = "0",
    ):
        self.input_dgw = input_dgw
        self.spectrum_dgw = spectrum_dgw
        self.model_uri = model_uri
        self.process_spectrum = ProcessSpectrumParameters(
            spectrum_dgw, overwrite_all_spectra
        )
        self.deploying = DeployModelParameters(
            redis_db,
            ion_mode="positive",
            overwrite_model=overwrite_model,
            environment=environment,
            pretrained=True,
        )


def build_pretrained_flow(
    project_name: str,
    flow_name: str,
    flow_config: FlowConfig,
    flow_parameters: PretrainedFlowParameters,
    mlflow_output_dir: str,
    mlflow_server: str,
    deploy_model: bool = False,
) -> Flow:
    """
    Builds a pretrained MS2DeepScore machine learning pipeline. It downloads a pre
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
    mlflow_output_dir:
        Directory for saving the model.
    mlflow_server:
        Server used for MLFlow to save the model.
    deploy_model:
        Whether to create a seldon deployment with the result of the training flow.
    Returns
    -------

    """
    with Flow(flow_name, **flow_config.kwargs) as training_flow:
        ms2deepscore_model_path = flow_parameters.model_uri

        ProcessSpectrum(flow_parameters.process_spectrum)()

        model_registry = RegisterModel(
            project_name,
            mlflow_output_dir,
            mlflow_server,
        )(ms2deepscore_model_path)

        if deploy_model:
            DeployModel(flow_parameters.deploying)(model_registry)

    return training_flow
