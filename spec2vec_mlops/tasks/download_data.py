import datetime

from prefect import task, Task

from spec2vec_mlops.tasks.data_gateway import InputDataGateway


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def download_data(
    uri: str, input_dgw: InputDataGateway, output_dir: str, dataset_id: str
) -> str:
    file_path = input_dgw.download_gnps(uri, output_dir, dataset_id)
    return file_path


class DownloadData(Task):
    def __init__(self, input_dgw: InputDataGateway, **kwargs):
        self._input_dgw = input_dgw
        super().__init__(
            max_retries=3, retry_delay=datetime.timedelta(seconds=10), **kwargs
        )

    def run(
        self, input_uri: str = None, output_dir: str = None, dataset_id: str = None
    ) -> str:
        file_path = self._input_dgw.download_gnps(input_uri, output_dir, dataset_id)
        return file_path
