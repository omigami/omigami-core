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
        download_path: str,
        dataset_name: str,
        result: Result,
        **kwargs,
    ):
        self._input_dgw = input_dgw
        self.input_uri = input_uri
        self.download_path = download_path

        config = {k: v for k, v in {**DEFAULT_CONFIG.copy(), **kwargs}.items()}

        super().__init__(
            **config,
            result=result,
            checkpoint=True,
            target=dataset_name,
        )

    def run(self) -> List[Dict[str, str]]:
        self._input_dgw.download_gnps(self.input_uri, self.download_path)
        gnps = self._input_dgw.load_gnps(self.download_path)
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
            download_path=self.download_path,
            input_dgw=self.input_dgw,
            dataset_name=self.dataset_name,
        )
