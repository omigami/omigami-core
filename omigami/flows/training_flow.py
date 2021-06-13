from dataclasses import dataclass
from typing import Union

from prefect import Flow, unmapped

from omigami.config import IonModes
from omigami.data_gateway import InputDataGateway, SpectrumDataGateway
from omigami.flows.config import FlowConfig
from omigami.flows.utils import create_result
from omigami.tasks import (
    DownloadData,
    MakeEmbeddings,
    DeployModel,
    DownloadParameters,
    CreateChunks,
    ChunkingParameters,
    ProcessSpectrum,
    TrainModel,
    TrainModelParameters,
    RegisterModel,
)
from omigami.tasks.process_spectrum import (
    ProcessSpectrumParameters,
)


@dataclass
class TrainingFlowParameters:
    input_dgw: InputDataGateway
    spectrum_dgw: SpectrumDataGateway
    downloading: DownloadParameters
    chunking: ChunkingParameters
    processing: ProcessSpectrumParameters
    training: TrainModelParameters


def build_training_flow(
    project_name: str,
    flow_parameters: TrainingFlowParameters,
    model_output_dir: str,
    mlflow_server: str,
    flow_config: FlowConfig,
    redis_db: str,
    ion_mode: IonModes = "positive",
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
    flow_name: str = "spec2vec-training-flow",
    deploy_model: bool = False,
) -> Flow:
    """
    Builds the spec2vec machine learning pipeline. It process data, trains a model, makes
    embeddings, registers the model and deploys it to the API.


    Parameters
    ----------
    project_name: str
        Prefect parameter. The project name.
    model_output_dir:
        Directory for saving the model.
    mlflow_server:
        Server used for MLFlow to save the model.
    ion_mode:
        The spectra ion mode used to train the model.
    intensity_weighting_power:
        Exponent used to scale intensity weights for each word.
    allowed_missing_percentage:
        Number of what percentage of a spectrum is allowed to be unknown to the model.
    flow_config: FlowConfig
        Configuration dataclass passed to prefect.Flow as a dict
    deploy_model:
        Whether to create a seldon deployment with the result of the training flow.
    Returns
    -------

    """
    with Flow(flow_name, **flow_config.kwargs) as training_flow:
        spectrum_ids = DownloadData(
            flow_parameters.input_dgw,
            flow_parameters.downloading,
            **create_result(flow_parameters.downloading.checkpoint_path),
        )()

        gnps_chunks = CreateChunks(
            flow_parameters.input_dgw,
            flow_parameters.chunking,
            **create_result(flow_parameters.chunking.chunk_paths_file),
        )(spectrum_ids)

        spectrum_ids_chunks = ProcessSpectrum(
            flow_parameters.input_dgw, flow_parameters.processing
        ).map(gnps_chunks)

        model = TrainModel(flow_parameters.spectrum_dgw, flow_parameters.training)(
            spectrum_ids_chunks
        )

        # TODO: add register model parameters
        model_registry = RegisterModel(
            project_name,
            model_output_dir,
            flow_parameters.processing.n_decimals,
            intensity_weighting_power,
            allowed_missing_percentage,
            mlflow_server,
        )(model)

        # TODO: this task prob doesnt need chunking or can be done in larger chunks
        _ = MakeEmbeddings(
            flow_parameters.spectrum_dgw,
            flow_parameters.processing.n_decimals,
            intensity_weighting_power,
            allowed_missing_percentage,
        ).map(unmapped(model), unmapped(model_registry), spectrum_ids_chunks)

        if deploy_model:
            DeployModel(redis_db)(model_registry)

    return training_flow
