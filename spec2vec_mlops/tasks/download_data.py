import datetime

from drfs import DRPath
from prefect import task

from spec2vec_mlops.helper_classes.data_downloader import DataDownloader


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def download_data_task(uri: str, out_dir: DRPath) -> DRPath:
    dl = DataDownloader(out_dir)
    file_path = dl.download_gnps_json(uri=uri)
    return DRPath(file_path)
