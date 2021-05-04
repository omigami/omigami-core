import logging
from typing import Union, Dict, Any

from prefect import Flow, unmapped, case

from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway
from spec2vec_mlops.tasks import (
    check_condition,
    DownloadData,
    LoadData,
    clean_data_task,
    train_model_task,
    register_model_task,
    make_embeddings_task,
    deploy_model_task,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def build_training_flow(
    project_name: str,
    source_uri: str,
    dataset_dir: str,
    dataset_id: str,
    model_output_dir: str,
    seldon_deployment_path: str,
    n_decimals: int,
    mlflow_server: str,
    iterations: int = 25,
    window: int = 500,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
    flow_config: Dict[str, Any] = None,
) -> Flow:
    """
    Function to register Prefect flow using remote cluster
    TODO: update

    Parameters
    ----------
    project_name
    source_uri
    dataset_dir
    dataset_id
    model_output_dir
    seldon_deployment_path
    n_decimals
    mlflow_server
    iterations:
        number of training iterations.
    window:
        window size for context words
    intensity_weighting_power:
        exponent used to scale intensity weights for each word
    allowed_missing_percentage:
        number of what percentage of a spectrum is allowed to be unknown to the model
    seldon_deployment_path:
        path to the seldon deployment configuration file
    flow_config

    Returns
    -------
    flow:
        The built flow
    """
    flow_config = flow_config or {}
    with Flow("spec2vec-training-flow", **flow_config) as training_flow:
        input_gtw = FSInputDataGateway()

        file_path = DownloadData(input_gtw)(source_uri, dataset_dir, dataset_id)
        input_data = LoadData(input_gtw)(file_path, chunk_size=20000)

        logger.info("Data loading is complete.")
        all_spectrum_ids_chunks = clean_data_task.map(
            input_data, n_decimals=unmapped(2), skip_if_exist=unmapped(True)
        )
        logger.info("Data cleaning and document conversion are complete.")

        with case(check_condition(all_spectrum_ids_chunks), True):
            model = train_model_task(iterations, window)
            run_id = register_model_task(
                mlflow_server,
                model,
                project_name,
                model_output_dir,
                n_decimals,
                intensity_weighting_power,
                allowed_missing_percentage,
            )
            logger.info("Model training is complete.")

        # TODO: this is make AND save embeddings. Prob need some refactor
        _ = make_embeddings_task.map(
            unmapped(model),
            all_spectrum_ids_chunks,
            unmapped(run_id),
            unmapped(n_decimals),
            unmapped(intensity_weighting_power),
            unmapped(allowed_missing_percentage),
        )
        logger.info("Saving embedding is complete.")
        deploy_model_task(run_id, seldon_deployment_path)

    return training_flow
