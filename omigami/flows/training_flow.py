import logging
from dataclasses import dataclass
from typing import Union

import prefect
from drfs import DRPath
from prefect import Flow, unmapped

from omigami.data_gateway import InputDataGateway
from omigami.flows.config import FlowConfig
from omigami.flows.utils import create_result
from omigami.tasks import (
    DownloadData,
    MakeEmbeddings,
    DeployModel,
    DownloadParameters,
    CreateChunks,
    ProcessSpectrum,
    TrainModel,
    TrainModelParameters,
    RegisterModel,
)
from omigami.tasks.process_spectrum import (
    ProcessSpectrumParameters,
)

logger = prefect.utilities.logging.get_logger()
logging.basicConfig(level=logging.DEBUG)


@dataclass  # WIP
class TrainingFlowParameters:
    download_params: DownloadParameters
    process_params: ProcessSpectrumParameters


def build_training_flow(
    project_name: str,
    input_dgw: InputDataGateway,
    download_params: DownloadParameters,
    process_params: ProcessSpectrumParameters,
    train_params: TrainModelParameters,
    model_output_dir: str,
    mlflow_server: str,
    flow_config: FlowConfig,
    redis_db: str,
    chunk_size: int = 1000000,
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
    download_params:
        Parameters of the DownloadData task.
    process_params:
        Parameters of the ProcessSpectrum task.
    train params:
        Parameters of the TrainModel task.
    model_output_dir:
        Diretory for saving the model.
    mlflow_server:
        Server used for MLFlow to save the model.
    chunk_size:
        Size of the chunks to map the data processing task.
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
            input_dgw,
            **download_params.kwargs,
            **create_result(download_params.checkpoint_path),
        )()

        # TODO: this looks ugly atm - create a dataclass for this task later
        gnps_chunks = CreateChunks(
            download_params.download_path,
            input_dgw,
            chunk_size,
            **create_result(
                f"{str(DRPath(download_params.download_path).parent)}/chunk_paths.pickle"
            ),
        )(spectrum_ids)

        # TODO: implement data caching like in DownloadData on s3
        spectrum_ids_chunks = ProcessSpectrum(
            download_params.download_path, input_dgw=input_dgw, **process_params.kwargs
        ).map(gnps_chunks)

        model = TrainModel(**train_params.kwargs)(spectrum_ids_chunks)

        model_registry = RegisterModel(
            project_name,
            model_output_dir,
            process_params.n_decimals,
            intensity_weighting_power,
            allowed_missing_percentage,
            mlflow_server,
        )(model)

        # TODO: this task prob doesnt need chunking or can be done in larger chunks
        _ = MakeEmbeddings(
            process_params.spectrum_dgw,
            process_params.n_decimals,
            intensity_weighting_power,
            allowed_missing_percentage,
        ).map(unmapped(model), unmapped(model_registry), spectrum_ids_chunks)

        logger.info("Saving embedding is complete.")
        if deploy_model:
            DeployModel(redis_db)(model_registry)

    return training_flow
