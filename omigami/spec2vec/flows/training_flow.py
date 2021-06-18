from typing import Union

from prefect import Flow, unmapped

from omigami.config import IonModes, ION_MODES
from omigami.data_gateway import InputDataGateway, SpectrumDataGateway
from omigami.spec2vec.flows.config import FlowConfig
from omigami.spec2vec.tasks import (
    DownloadData,
    MakeEmbeddings,
    DeployModel,
    DeployModelParameters,
    DownloadParameters,
    CreateChunks,
    ChunkingParameters,
    ProcessSpectrum,
    TrainModel,
    TrainModelParameters,
    RegisterModel,
)
from omigami.spec2vec.tasks.process_spectrum import (
    ProcessSpectrumParameters,
)


class TrainingFlowParameters:
    def __init__(
        self,
        input_dgw: InputDataGateway,
        spectrum_dgw: SpectrumDataGateway,
        source_uri: str,
        output_dir: str,
        dataset_id: str,
        chunk_size: int,
        ion_mode: IonModes,
        n_decimals: int,
        skip_if_exists: bool,
        iterations: int,
        window: int,
        dataset_name: str = "gnps.json",
        dataset_checkpoint_name: str = "spectrum_ids.pkl",
        redis_db: str = "0",
        overwrite: bool = False,
        environment: str = "dev",
    ):
        self.input_dgw = input_dgw
        self.spectrum_dgw = spectrum_dgw

        if ion_mode not in ION_MODES:
            raise ValueError("Ion mode can only be either 'positive' or 'negative'.")

        self.downloading = DownloadParameters(
            source_uri, output_dir, dataset_id, dataset_name, dataset_checkpoint_name
        )
        self.chunking = ChunkingParameters(
            self.downloading.download_path, chunk_size, ion_mode
        )
        self.processing = ProcessSpectrumParameters(
            spectrum_dgw,
            n_decimals,
            skip_if_exists,
        )
        self.training = TrainModelParameters(iterations, window)
        self.deploying = DeployModelParameters(
            redis_db, ion_mode, overwrite, environment
        )


def build_training_flow(
    project_name: str,
    flow_name: str,
    flow_config: FlowConfig,
    flow_parameters: TrainingFlowParameters,
    # TODO: incorporate the next parameters into data classes
    model_output_dir: str,
    mlflow_server: str,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
    deploy_model: bool = False,
) -> Flow:
    """
    Builds the spec2vec machine learning pipeline. It process data, trains a model, makes
    embeddings, registers the model and deploys it to the API.


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
    model_output_dir:
        Directory for saving the model.
    mlflow_server:
        Server used for MLFlow to save the model.
    intensity_weighting_power:
        Exponent used to scale intensity weights for each word.
    allowed_missing_percentage:
        Number of what percentage of a spectrum is allowed to be unknown to the model.
    deploy_model:
        Whether to create a seldon deployment with the result of the training flow.
    Returns
    -------

    """
    with Flow(flow_name, **flow_config.kwargs) as training_flow:
        spectrum_ids = DownloadData(
            flow_parameters.input_dgw,
            flow_parameters.downloading,
        )()

        gnps_chunks = CreateChunks(
            flow_parameters.input_dgw,
            flow_parameters.chunking,
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
            DeployModel(flow_parameters.deploying)(model_registry)

    return training_flow
