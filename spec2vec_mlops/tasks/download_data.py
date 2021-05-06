from typing import Dict, List

from prefect import Task
from prefect.engine.result import Result

from spec2vec_mlops.tasks.config import DEFAULT_CONFIG
from spec2vec_mlops.tasks.data_gateway import InputDataGateway


class DownloadData(Task):
    def __init__(self, input_dgw: InputDataGateway, result: Result, target: str):
        self._input_dgw = input_dgw

        super().__init__(
            **DEFAULT_CONFIG,
            result=result,
            checkpoint=True,
            target=target,
        )

    def run(
        self, input_uri: str = None, dataset_dir: str = None
    ) -> List[Dict[str, str]]:
        file_path = self._input_dgw.download_gnps(input_uri, dataset_dir)
        gnps = self._input_dgw.load_gnps(file_path)
        return gnps
