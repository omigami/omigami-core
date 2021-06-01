import logging
from dataclasses import dataclass
from typing import Union

from prefect import Flow, unmapped
import prefect

from omigami.data_gateway import SpectrumDataGateway
from omigami.flows.config import FlowConfig
from omigami.flows.utils import create_result
from omigami.tasks import (
    DownloadData,
    DownloadParameters,
    CreateChunks,
    ProcessSpectrum,
    ProcessSpectrumParameters,
    TrainModel,
    TrainModelParameters,
    MakeEmbeddings,
    DeployModel,
)

logger = prefect.utilities.logging.get_logger()
logging.basicConfig(level=logging.DEBUG)


@dataclass  # WIP
class TrainingFlowParameters:
    download_params: DownloadParameters
    process_params: ProcessSpectrumParameters


def build_training_flow(
    download_params: DownloadParameters,
    process_params: ProcessSpectrumParameters,
    training_params: TrainModelParameters,
    spectrum_dgw: SpectrumDataGateway,
    flow_config: FlowConfig,
    redis_db: str,
    chunk_size: int = 1000,
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
    download_params:
        Parameters of the DownloadData task.
    process_params:
        Parameters of the ProcessSpectrum task
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
            **download_params.kwargs,
            **create_result(download_params.checkpoint_path),
        )()

        spectrum_id_chunks = CreateChunks(chunk_size)(spectrum_ids)

        # TODO: implement data caching like in DownloadData on s3
        all_spectrum_ids_chunks = ProcessSpectrum(
            download_params.download_path,
            spectrum_dgw,
            **process_params.kwargs,
        ).map(spectrum_id_chunks)

        model_registry = TrainModel(training_params, spectrum_dgw)()

        # TODO: this task prob doesnt need chunking or can be done in larger chunks
        _ = MakeEmbeddings(
            process_params.spectrum_dgw,
            process_params.n_decimals,
            intensity_weighting_power,
            allowed_missing_percentage,
        ).map(unmapped(model_registry), all_spectrum_ids_chunks)
        logger.info("Saving embedding is complete.")

        if deploy_model:
            DeployModel(redis_db)(model_registry)

    return training_flow
