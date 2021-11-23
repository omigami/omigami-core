from prefect import Flow

from omigami.flow_config import FlowConfig
from omigami.gateways.data_gateway import DataGateway
from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.ms2deepscore.tasks import (
    RegisterModel,
    ProcessSpectrumParameters,
    ProcessSpectrum,
    RegisterModelParameters,
)
from omigami.tasks import DeployModel, DeployModelParameters


class PretrainedFlowParameters:
    def __init__(
        self,
        data_gtw: DataGateway,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        model_uri: str,
        spectrum_binner_output_path: str,
        project_name: str,
        mlflow_output_dir: str,
        mlflow_server: str,
        overwrite_model: bool = False,
        overwrite_all_spectra: bool = False,
        spectrum_binner_n_bins: int = 10000,
        redis_db: str = "0",
    ):
        self.data_gtw = data_gtw
        self.spectrum_dgw = spectrum_dgw
        self.model_uri = model_uri
        self.process_spectrum = ProcessSpectrumParameters(
            spectrum_binner_output_path,
            ion_mode="positive",
            overwrite_all_spectra=overwrite_all_spectra,
            is_pretrained_flow=True,
            n_bins=spectrum_binner_n_bins,
        )
        self.registering = RegisterModelParameters(
            project_name, mlflow_output_dir, mlflow_server, "positive"
        )
        self.deploying = DeployModelParameters(
            redis_db, overwrite_model, "ms2deep-pretrained"
        )


def build_pretrained_flow(
    flow_name: str,
    flow_config: FlowConfig,
    flow_parameters: PretrainedFlowParameters,
    deploy_model: bool = False,
) -> Flow:
    """
    Builds a pretrained MS2DeepScore machine learning pipeline. It downloads a pre
    trained model, registers it and deploys it to the API.


    Parameters
    ----------
    flow_name:
        Name of the flow
    flow_config: FlowConfig
        Configuration dataclass passed to prefect.Flow as a dict
    flow_parameters:
        Class containing all flow parameters
    deploy_model:
        Whether to create a seldon deployment with the result of the training flow.
    Returns
    -------

    """
    with Flow(flow_name, **flow_config.kwargs) as training_flow:
        ms2deepscore_model_path = flow_parameters.model_uri

        ProcessSpectrum(
            flow_parameters.data_gtw,
            flow_parameters.spectrum_dgw,
            flow_parameters.process_spectrum,
        )()

        model_registry = RegisterModel(flow_parameters.registering)(
            {
                "ms2deepscore_model_path": ms2deepscore_model_path,
                "validation_loss": None,
            }
        )

        if deploy_model:
            DeployModel(flow_parameters.deploying)(model_registry)

    return training_flow
