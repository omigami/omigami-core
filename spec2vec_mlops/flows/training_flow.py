import logging
from typing import Union, Dict, Any

from prefect import Flow, unmapped, case

from spec2vec_mlops.tasks import (
    check_condition,
    DownloadData,
    LoadData,
    train_model_task,
    register_model_task,
    make_embeddings_task,
    deploy_model_task,
)
from spec2vec_mlops.tasks.clean_data import CleanData
from spec2vec_mlops.tasks.data_gateway import InputDataGateway, SpectrumDataGateway

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
    skip_if_exists: bool,
    mlflow_server: str,
    input_dgw: InputDataGateway,
    spectrum_gtw: SpectrumDataGateway,
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
        file_path = DownloadData(input_dgw)(source_uri, dataset_dir, dataset_id)
        input_data = LoadData(input_dgw)(file_path, chunk_size=20000)

        logger.info("Started data cleaning and conversion to documents.")
        all_spectrum_ids_chunks = CleanData(
            spectrum_gtw, n_decimals, skip_if_exists
        ).map(input_data)

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
