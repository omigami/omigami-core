from datetime import timedelta, date, datetime
from typing import Optional

from prefect import Flow, unmapped
from prefect.schedules import Schedule
from prefect.schedules.clocks import IntervalClock

from omigami.config import IonModes, ION_MODES
from omigami.flow_config import FlowConfig
from omigami.gateways.data_gateway import DataGateway
from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.ms2deepscore.helper_classes.siamese_model_trainer import SplitRatio
from omigami.ms2deepscore.tasks import (
    ProcessSpectrumParameters,
    ProcessSpectrum,
    RegisterModel,
    RegisterModelParameters,
    MakeEmbeddingsParameters,
    MakeEmbeddings,
    CreateSpectrumIDsChunks,
    ChunkingIDsParameters,
)
from omigami.ms2deepscore.tasks.calculate_tanimoto_score import (
    CalculateTanimotoScoreParameters,
    CalculateTanimotoScore,
)
from omigami.ms2deepscore.tasks.train_model import TrainModelParameters, TrainModel
from omigami.spectrum_cleaner import SpectrumCleaner
from omigami.tasks import (
    DownloadParameters,
    DownloadData,
    ChunkingParameters,
    CreateChunks,
    SaveRawSpectraParameters,
    SaveRawSpectra,
    DeployModelParameters,
    DeployModel,
)


class TrainingFlowParameters:
    def __init__(
        self,
        data_gtw: DataGateway,
        spectrum_dgw: MS2DeepScoreRedisSpectrumDataGateway,
        spectrum_cleaner: SpectrumCleaner,
        source_uri: str,
        dataset_directory: str,
        dataset_id: str,
        chunk_size: int,
        ion_mode: IonModes,
        scores_output_path: str,
        fingerprint_n_bits: int,
        scores_decimals: int,
        overwrite_all_spectra: bool,
        spectrum_binner_output_path: str,
        spectrum_binner_n_bins: int,
        overwrite_model: bool,
        model_output_path: str,
        project_name: str,
        mlflow_output_directory: str,
        mlflow_server: str,
        epochs: int = 50,
        train_ratio: float = 0.9,
        validation_ratio: float = 0.05,
        test_ratio: float = 0.05,
        spectrum_ids_chunk_size: int = 10000,
        schedule_task_days: Optional[int] = 30,
        dataset_name: str = "gnps.json",
        dataset_checkpoint_name: str = "spectrum_ids.pkl",
        redis_db: str = "0",
    ):
        self.data_gtw = data_gtw
        self.spectrum_dgw = spectrum_dgw

        if schedule_task_days != None:
            self.schedule = Schedule(
                clocks=[
                    IntervalClock(
                        start_date=datetime.combine(date.today(), datetime.min.time()),
                        interval=timedelta(days=schedule_task_days),
                    )
                ]
            )
        else:
            self.schedule = None

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
            spectrum_dgw, data_gtw, spectrum_cleaner, overwrite_all_spectra
        )

        self.process_spectrum = ProcessSpectrumParameters(
            spectrum_binner_output_path,
            ion_mode=ion_mode,
            n_bins=spectrum_binner_n_bins,
        )

        self.calculate_tanimoto_score = CalculateTanimotoScoreParameters(
            scores_output_path, ion_mode, fingerprint_n_bits, scores_decimals
        )

        self.training = TrainModelParameters(
            model_output_path,
            ion_mode,
            spectrum_binner_output_path,
            epochs,
            SplitRatio(train_ratio, validation_ratio, test_ratio),
        )

        self.registering = RegisterModelParameters(
            project_name, mlflow_output_directory, mlflow_server, ion_mode
        )

        self.spectrum_chunking = ChunkingIDsParameters(spectrum_ids_chunk_size)

        self.embedding = MakeEmbeddingsParameters(ion_mode)

        self.deploying = DeployModelParameters(
            redis_db, overwrite_model, f"ms2deepscore-{ion_mode}"
        )


def build_training_flow(
    flow_name: str,
    flow_config: FlowConfig,
    flow_parameters: TrainingFlowParameters,
    deploy_model: bool = False,
) -> Flow:
    """
    Builds the MS2DeepScore machine learning pipeline. It Downloads and process data, trains the model, makes
    embeddings, registers the model and deploys it to the API.


    Parameters
    ----------
    flow_name:
        Name of the flow
    flow_config: FlowConfig
        Configuration dataclass passed to prefect.Flow as a dict
    flow_parameters:
        Dataclass containing all flow parameters
    deploy_model:
        If the model will be deployed or not
    -------

    """

    with Flow(name=flow_name, **flow_config.kwargs) as training_flow:
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

        spectrum_ids_chunks = SaveRawSpectra(flow_parameters.save_raw_spectra).map(
            gnps_chunks
        )

        processed_ids = ProcessSpectrum(
            flow_parameters.data_gtw,
            flow_parameters.spectrum_dgw,
            flow_parameters.process_spectrum,
        )(spectrum_ids_chunks)

        scores_output_path = CalculateTanimotoScore(
            flow_parameters.spectrum_dgw, flow_parameters.calculate_tanimoto_score
        )(processed_ids)

        train_model_output = TrainModel(
            flow_parameters.data_gtw,
            flow_parameters.spectrum_dgw,
            flow_parameters.training,
            nout=2,
        )(processed_ids, scores_output_path)

        model_registry = RegisterModel(
            flow_parameters.registering, training_parameters=flow_parameters.training
        )(train_model_output)

        processed_chunks = CreateSpectrumIDsChunks(flow_parameters.spectrum_chunking)(
            processed_ids
        )

        MakeEmbeddings(
            flow_parameters.spectrum_dgw,
            flow_parameters.data_gtw,
            flow_parameters.embedding,
        ).map(
            unmapped(train_model_output),
            unmapped(model_registry),
            processed_chunks,
        )

        if deploy_model:
            DeployModel(flow_parameters.deploying)(model_registry)

    return training_flow
