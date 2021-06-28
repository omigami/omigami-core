from dataclasses import dataclass

from prefect import Task

from omigami.utils import merge_prefect_task_configs
from omigami.gateways.data_gateway import InputDataGateway
from omigami.utils import create_prefect_result_from_path


@dataclass
class DownloadPreTrainedModelParameters:
    model_uri: str
    output_dir: str
    file_name: str = "ms2deep_score.hdf5"

    @property
    def output_path(self):
        return f"{self.output_dir}/{self.file_name}"


class DownloadPreTrainedModel(Task):
    """
    Prefect task to download pre-trained ms2deep score model from url
    """

    def __init__(
        self,
        input_datagateway: InputDataGateway,
        download_parameters: DownloadPreTrainedModelParameters,
        **kwargs,
    ):
        self._input_datagateway = input_datagateway
        self.model_uri = download_parameters.model_uri
        self.output_path = download_parameters.output_path

        config = merge_prefect_task_configs(kwargs)

        super().__init__(
            **config,
            **create_prefect_result_from_path(self.output_path),
            checkpoint=True,
        )

    def run(self) -> str:
        self.logger.info(f"Downloading pre-trained MS2Deep model from {self.model_uri}")
        self._input_datagateway.download_ms2deep_model(self.model_uri, self.output_path)
        self.logger.info(f"Saving pre-trained MS2Deep model to {self.output_path}")

        return self.output_path
