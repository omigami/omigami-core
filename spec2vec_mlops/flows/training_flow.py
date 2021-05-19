import logging
from dataclasses import dataclass
from typing import Union

from prefect import Flow, unmapped, case

from spec2vec_mlops.flows.config import FlowConfig
from spec2vec_mlops.flows.utils import create_result
from spec2vec_mlops.tasks import (
    check_condition,
    DownloadData,
    train_model_task,
    register_model_task,
    make_embeddings_task,
    deploy_model_task,
)
from spec2vec_mlops.tasks.process_spectrum import (
    ProcessSpectrum,
    ProcessSpectrumParameters,
)
from spec2vec_mlops.tasks.download_data import DownloadParameters
from spec2vec_mlops.tasks.process_spectrum.create_chunks import CreateChunks

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass  # WIP
class TrainingFlowParameters:
    download_params: DownloadParameters
    process_params: ProcessSpectrumParameters


def build_training_flow(
    project_name: str,
    download_params: DownloadParameters,
    process_params: ProcessSpectrumParameters,
    model_output_dir: str,
    mlflow_server: str,
    flow_config: FlowConfig,
    redis_db: str,
    chunk_size: int = 1000,
    iterations: int = 25,
    window: int = 500,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
) -> Flow:
    """
    Builds the spec2vec machine learning pipeline. It process data, trains a model, makes
    embeddings, registers the model and deploys it to the API.


    Parameters
    ----------
    project_name: str
        Prefect parameter. The project name.
    download_params:
        Parameters of the DownloadData task
    process_params:
        Parameters of the ProcessSpectrum task
    model_output_dir:
        Diretory for saving the model
    mlflow_server:
        Server used for MLFlow to save the model
    chunk_size:
        Size of the chunks to map the data processing task
    iterations:
        Number of training iterations
    window:
        Window size for context around the word
    intensity_weighting_power:
        Exponent used to scale intensity weights for each word
    allowed_missing_percentage:
        Number of what percentage of a spectrum is allowed to be unknown to the model
    flow_config: FlowConfig
        Configuration dataclass passed to prefect.Flow as a dict

    Returns
    -------

    """
    with Flow("spec2vec-training-flow", **flow_config.kwargs) as training_flow:
        logger.info("Downloading and loading spectrum data.")
        spectrum_ids = DownloadData(
            **download_params.kwargs,
            result=create_result(download_params.checkpoint_path),
        )()

        spectrum_id_chunks = CreateChunks(chunk_size)(spectrum_ids)

        logger.info("Started data cleaning and conversion to documents.")
        # TODO: implement data caching like in DownloadData here. Will need to implement
        # TODO: a new class like RedisResult
        all_spectrum_ids_chunks = ProcessSpectrum(
            download_params.download_path, **process_params.kwargs
        ).map(spectrum_id_chunks)

        # TODO: this case can be removed if we link train with clean data via input/output
        with case(check_condition(all_spectrum_ids_chunks), True):
            model = train_model_task(iterations, window)
            registered_model = register_model_task(
                mlflow_server,
                model,
                project_name,
                model_output_dir,
                process_params.n_decimals,
                intensity_weighting_power,
                allowed_missing_percentage,
            )
            logger.info("Model training is complete.")

        # TODO: this is make AND save embeddings. Prob need some refactor
        _ = make_embeddings_task.map(
            unmapped(model),
            all_spectrum_ids_chunks,
            unmapped(registered_model),
            unmapped(process_params.n_decimals),
            unmapped(intensity_weighting_power),
            unmapped(allowed_missing_percentage),
        )
        logger.info("Saving embedding is complete.")
        # deploy_model_task(registered_model, redis_db)

    return training_flow
