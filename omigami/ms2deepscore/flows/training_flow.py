from dataclasses import dataclass

from typing import Union

from prefect import Flow, unmapped
from prefect.schedules import Schedule
from prefect.schedules.clocks import IntervalClock

from omigami.config import IonModes, ION_MODES
from omigami.gateways.data_gateway import InputDataGateway, SpectrumDataGateway
from omigami.flow_config import FlowConfig
from omigami.shared_tasks import DownloadData, DownloadParameters

from datetime import timedelta, date, datetime


class TrainingFlowParameters:
    def __init__(
        self,
        input_dgw: InputDataGateway,
        spectrum_dgw: SpectrumDataGateway,
        source_uri: str,
        output_dir: str,
        dataset_id: str,
        ion_mode: IonModes,
        schedule_task_days: int = 30,
        dataset_name: str = "gnps.json",
        dataset_checkpoint_name: str = "spectrum_ids.pkl",
    ):
        self.input_dgw = input_dgw
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
            source_uri, output_dir, dataset_id, dataset_name, dataset_checkpoint_name
        )


@dataclass
class ModelGeneralParameters:
    model_output_dir: str
    mlflow_server: str
    deploy_model: bool = False


def build_training_flow(
    project_name: str,
    flow_name: str,
    flow_config: FlowConfig,
    flow_parameters: TrainingFlowParameters,
    model_parameters: ModelGeneralParameters,
) -> Flow:
    """
    Builds the MS2DeepScore machine learning pipeline. It Downloads and process data, trains the model, makes
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
        Dataclass containing all flow parameters
    model_parameters:
        Dataclass containing general information for how to handle the model, like the output directory.
    -------

    """

    with Flow(name=flow_name, **flow_config.kwargs) as training_flow:
        # TODO: Do we need a check here to see if the dataset is actually older then 30 days?

        spectrum_ids = DownloadData(
            flow_parameters.input_dgw,
            flow_parameters.downloading,
        )()

    return training_flow
