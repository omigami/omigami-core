import logging
from dataclasses import dataclass
from typing import Union, Dict, Any

import prefect
from prefect import Flow, case, unmapped

from spec2vec_mlops.flows.utils import create_result
from spec2vec_mlops.tasks import (
    check_condition,
    DownloadData,
    MakeEmbeddings,
    ProcessSpectrum,
    train_model_task,
    register_model_task,
    deploy_model_task,
)
from spec2vec_mlops.tasks.process_spectrum import (
    ProcessSpectrumParameters,
)
from spec2vec_mlops.tasks.download_data import DownloadParameters
from spec2vec_mlops.tasks.process_spectrum.create_chunks import CreateChunks

logger = prefect.utilities.logging.get_logger()
logging.basicConfig(level=logging.DEBUG)


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
    chunk_size: int = 1000,
    iterations: int = 25,
    window: int = 500,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
    flow_config: Dict[str, Any] = None,
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
    flow_config:
        Configuration passed to prefect.Flow
    deploy_model:
        Whether to create a seldon deployment with the result of the training flow

    Returns
    -------

    """
    flow_config = flow_config or {}
    with Flow(flow_name, **flow_config) as training_flow:
        logger.info(
            f"Downloading and loading spectrum data from {download_params.input_uri} to "
            f"{download_params.download_path}."
        )
        spectrum_ids = DownloadData(
            **download_params.kwargs,
            **create_result(download_params.checkpoint_path),
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
            model_registry = register_model_task(
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
        _ = MakeEmbeddings(
            process_params.spectrum_dgw,
            process_params.n_decimals,
            intensity_weighting_power,
            allowed_missing_percentage,
        ).map(unmapped(model), unmapped(model_registry), all_spectrum_ids_chunks)
        logger.info("Saving embedding is complete.")
        if deploy_model:
            deploy_model_task(model_registry)

    return training_flow
