from typing import Union, Optional

from prefect import Flow, unmapped

from omigami.config import IonModes, ION_MODES
from omigami.flow_config import FlowConfig
from omigami.spectra_matching.gateways import RedisSpectrumDataGateway
from omigami.spectra_matching.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.gateways.spectrum_document import SpectrumDocumentDataGateway
from omigami.spec2vec.tasks import (
    MakeEmbeddings,
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
from omigami.spectra_matching.tasks import (
    DownloadData,
    DownloadParameters,
    ChunkingParameters,
    CreateChunks,
    SaveRawSpectra,
    SaveRawSpectraParameters,
    DeployModelParameters,
    DeployModel,
)


class TrainingFlowParameters:
    def __init__(
        self,
        spectrum_dgw: RedisSpectrumDataGateway,
        data_gtw: FSDataGateway,
        document_dgw: SpectrumDocumentDataGateway,
        spectrum_cleaner: SpectrumCleaner,
        source_uri: str,
        dataset_directory: str,
        dataset_id: str,
        chunk_size: int,
        ion_mode: IonModes,
        n_decimals: int,
        overwrite_all_spectra: bool,
        iterations: int,
        window: int,
        mlflow_output_directory: str,
        documents_save_directory: str,
        mlflow_server: str,
        intensity_weighting_power: Union[float, int] = 0.5,
        allowed_missing_percentage: Union[float, int] = 5.0,
        dataset_name: str = "gnps.json",
        dataset_checkpoint_name: str = "spectrum_ids.pkl",
        redis_db: str = "0",
        overwrite_model: bool = False,
        model_name: Optional[str] = "spec2vec-model",
        experiment_name: str = "default",
    ):
        self.data_gtw = data_gtw
        self.spectrum_dgw = spectrum_dgw
        self.document_dgw = document_dgw
        if ion_mode not in ION_MODES:
            raise ValueError("Ion mode can only be either 'positive' or 'negative'.")

        self.downloading = DownloadParameters(
            source_uri,
            dataset_directory,
            dataset_id,
            dataset_name,
            dataset_checkpoint_name,
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
            experiment_name=experiment_name,
            mlflow_output_directory=mlflow_output_directory,
            server_uri=mlflow_server,
            n_decimals=n_decimals,
            ion_mode=ion_mode,
            intensity_weighting_power=intensity_weighting_power,
            allowed_missing_percentage=allowed_missing_percentage,
            model_name=model_name,
        )
        self.embedding = MakeEmbeddingsParameters(
            ion_mode, n_decimals, intensity_weighting_power, allowed_missing_percentage
        )
        self.deploying = DeployModelParameters(
            redis_db, overwrite_model, model_name=f"spec2vec-{ion_mode}"
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
        # tasks return objects, so that we could connect tasks using returned objects.
        # if you update one return type of a task please mind the later tasks too

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
            flow_parameters.document_dgw,
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
