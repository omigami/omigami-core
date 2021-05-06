import logging
from dataclasses import dataclass
from typing import Union, Dict, Any

from prefect import Flow, unmapped, case

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
from spec2vec_mlops.tasks.create_chunks import CreateChunks
from spec2vec_mlops.tasks.download_data import DownloadParameters

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass  # WIP
class TrainingFlowParameters:
    download_params: DownloadParameters
    process_params: ProcessSpectrumParameters


# TODO: maybe this should be a class so this interface is not so huge? or add dataclasses
# TODO: to reduce parameters.
def build_training_flow(
    project_name: str,
    download_params: DownloadParameters,
    process_params: ProcessSpectrumParameters,
    model_output_dir: str,
    seldon_deployment_path: str,
    mlflow_server: str,
    chunk_size: int = 1000,
    iterations: int = 25,
    window: int = 500,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
    flow_config: Dict[str, Any] = None,
) -> Flow:
    """
    TODO: update

    Returns
    -------
    flow:
        The built flow
    """
    flow_config = flow_config or {}
    with Flow("spec2vec-training-flow", **flow_config) as training_flow:
        logger.info("Downloading and loading spectrum data.")
        spectrum = DownloadData(
            **download_params.kwargs,
            result=create_result(download_params.download_path),
        )()

        chunked_spectrum = CreateChunks(chunk_size)(spectrum)

        logger.info("Started data cleaning and conversion to documents.")
        # TODO: implement data caching like in DownloadData here. Will need to implement
        # TODO: a new class like RedisResult
        all_spectrum_ids_chunks = ProcessSpectrum(**process_params.kwargs).map(
            chunked_spectrum
        )

        # TODO: this case can be removed if we link train with clean data via input/output
        with case(check_condition(all_spectrum_ids_chunks), True):
            model = train_model_task(iterations, window)
            run_id = register_model_task(
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
            unmapped(run_id),
            unmapped(process_params.n_decimals),
            unmapped(intensity_weighting_power),
            unmapped(allowed_missing_percentage),
        )
        logger.info("Saving embedding is complete.")
        deploy_model_task(run_id, seldon_deployment_path)

    return training_flow
