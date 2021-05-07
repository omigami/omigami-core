from dataclasses import dataclass
from typing import Dict, List

from prefect import Task
from prefect.engine.result import Result

from spec2vec_mlops.tasks.config import DEFAULT_CONFIG
from spec2vec_mlops.tasks.data_gateway import InputDataGateway


class DownloadData(Task):
    def __init__(
        self,
        input_dgw: InputDataGateway,
        input_uri: str,
        output_path: str,
        dataset_name: str,
        result: Result,
    ):
        self._input_dgw = input_dgw
        self.input_uri = input_uri
        self.output_path = output_path

        super().__init__(
            **DEFAULT_CONFIG,
            result=result,
            checkpoint=True,
            target=dataset_name,
        )

    def run(self) -> List[Dict[str, str]]:
        self._input_dgw.download_gnps(self.input_uri, self.output_path)
        gnps = self._input_dgw.load_gnps(self.output_path)
        return gnps


@dataclass
class DownloadParameters:
    input_uri: str
    output_dir: str
    dataset_name: str
    input_dgw: InputDataGateway

    @property
    def download_path(self):
        return f"{self.output_dir}/{self.dataset_name}"

    @property
    def kwargs(self):
        return dict(
            input_uri=self.input_uri,
            output_dir=self.output_dir,
            input_dgw=self.input_dgw,
            dataset_name=self.dataset_name,
        )
