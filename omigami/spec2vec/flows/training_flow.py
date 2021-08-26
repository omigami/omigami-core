from typing import Union

from omigami.config import IonModes, ION_MODES
from omigami.flow_config import FlowConfig
from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.gateways.gateway_controller import Spec2VecGatewayController

from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)

from omigami.spec2vec.tasks import (
    MakeEmbeddings,
    DeployModel,
    DeployModelParameters,
    ProcessSpectrum,
    TrainModel,
    TrainModelParameters,
    RegisterModel,
    RegisterModelParameters,
    MakeEmbeddingsParameters,
)
from omigami.spec2vec.tasks.process_spectrum import (
    ProcessSpectrumParameters,
)
from omigami.spectrum_cleaner import SpectrumCleaner
from omigami.tasks import (
    DownloadData,
    DownloadParameters,
    ChunkingParameters,
    CreateChunks,
    SaveRawSpectra,
    SaveRawSpectraParameters,
)
from prefect import Flow, unmapped


class TrainingFlowParameters:
    def __init__(
        self,
        spectrum_dgw: Spec2VecRedisSpectrumDataGateway,
        data_gtw: FSDataGateway,
        document_dgw_controller: Spec2VecGatewayController,
        spectrum_cleaner: SpectrumCleaner,
        source_uri: str,
        output_dir: str,
        dataset_id: str,
        chunk_size: int,
        ion_mode: IonModes,
        n_decimals: int,
        overwrite_all_spectra: bool,
        iterations: int,
        window: int,
        project_name: str,
        model_output_dir: str,
        documents_save_directory: str,
        mlflow_server: str,
        intensity_weighting_power: Union[float, int] = 0.5,
        allowed_missing_percentage: Union[float, int] = 5.0,
        dataset_name: str = "gnps.json",
        dataset_checkpoint_name: str = "spectrum_ids.pkl",
        redis_db: str = "0",
        overwrite_model: bool = False,
        environment: str = "dev",
    ):
        self.data_gtw = data_gtw
        self.spectrum_dgw = spectrum_dgw
        self.document_dgw_controller = document_dgw_controller
        if ion_mode not in ION_MODES:
            raise ValueError("Ion mode can only be either 'positive' or 'negative'.")

        self.downloading = DownloadParameters(
            source_uri, output_dir, dataset_id, dataset_name, dataset_checkpoint_name
        )
        self.chunking = ChunkingParameters(
            self.downloading.download_path, chunk_size, ion_mode
        )
        self.save_raw_spectra = SaveRawSpectraParameters(
            spectrum_dgw, data_gtw, spectrum_cleaner
        )
        self.processing = ProcessSpectrumParameters(
            spectrum_dgw,
            documents_save_directory,
            ion_mode,
            n_decimals,
            overwrite_all_spectra,
        )
        self.training = TrainModelParameters(iterations, window)
        self.registering = RegisterModelParameters(
            project_name,
            model_output_dir,
            mlflow_server,
            n_decimals,
            ion_mode,
            intensity_weighting_power,
            allowed_missing_percentage,
        )
        self.embedding = MakeEmbeddingsParameters(
            ion_mode, n_decimals, intensity_weighting_power, allowed_missing_percentage
        )
        self.deploying = DeployModelParameters(
            redis_db, ion_mode, overwrite_model, environment
        )


def build_training_flow(
    flow_name: str,
    flow_config: FlowConfig,
    flow_parameters: TrainingFlowParameters,
    deploy_model: bool = False,
) -> Flow:
    """
    Builds the spec2vec machine learning pipeline. It process data, trains a model, makes
    embeddings, registers the model and deploys it to the API.


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
        spectrum_ids = DownloadData(
            flow_parameters.data_gtw,
            flow_parameters.downloading,
        )()

        gnps_chunks = CreateChunks(
            flow_parameters.data_gtw,
            flow_parameters.chunking,
        )(spectrum_ids)

        chunked_spectrum_ids = SaveRawSpectra(flow_parameters.save_raw_spectra).map(
            gnps_chunks
        )

        document_paths = ProcessSpectrum(
            flow_parameters.data_gtw,
            flow_parameters.document_dgw_controller,
            flow_parameters.processing,
        ).map(chunked_spectrum_ids)

        model = TrainModel(flow_parameters.data_gtw, flow_parameters.training)(
            document_paths
        )

        model_registry = RegisterModel(flow_parameters.registering)(model)

        # TODO: this task prob doesnt need chunking or can be done in larger chunks
        _ = MakeEmbeddings(
            flow_parameters.spectrum_dgw,
            flow_parameters.data_gtw,
            flow_parameters.embedding,
        ).map(unmapped(model), unmapped(model_registry), document_paths)

        if deploy_model:
            DeployModel(flow_parameters.deploying)(model_registry)

    return training_flow
