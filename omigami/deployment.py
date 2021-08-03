from datetime import timedelta
from typing import Optional

from omigami.authentication.authenticator import KratosAuthenticator
from omigami.config import (
    API_SERVER_URLS,
    MLFLOW_SERVER,
    REDIS_DATABASES,
    DATASET_IDS,
    IonModes,
    SOURCE_URI_PARTIAL_GNPS,
)
from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from prefect import Client
from typing_extensions import Literal


class Deployer:
    def __init__(
        self,
        image: str,
        dataset_name: str,
        mlflow_server: str = MLFLOW_SERVER,
        source_uri: str = SOURCE_URI_PARTIAL_GNPS,
        environment: Literal["dev", "prod"] = "dev",
        chunk_size: int = int(1e8),
        overwrite_model: bool = False,
        overwrite_all_spectra: bool = True,
        auth: bool = False,
        auth_url: Optional[str] = None,
        ion_mode: IonModes = "positive",
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_server: Optional[str] = None,
        session_token: Optional[str] = None,
        deploy_model: bool = False,
        schedule: timedelta = None,
    ):
        self._image = image
        self._mlflow_server = mlflow_server
        self._auth = auth
        self._ion_mode = ion_mode
        self._deploy_model = deploy_model
        self._schedule = schedule
        self._overwrite_model = overwrite_model
        self._overwrite_all_spectra = overwrite_all_spectra
        self._source_uri = source_uri
        self._chunk_size = chunk_size

        # validates parameters and use them to get task configuration variables
        if environment not in ["dev", "prod"]:
            raise ValueError("Environment not valid. Should be either 'dev' or 'prod'.")

        if dataset_name not in DATASET_IDS[environment].keys():
            raise ValueError(
                f"No such option available for reference dataset: {dataset_name}. Available options are:"
                f"{list(DATASET_IDS[environment].keys())}."
            )

        self._environment = environment
        self._dataset_name = dataset_name
        self._auth_url = auth_url
        self._username = username
        self._password = password
        self._api_server = api_server
        self._redis_db = REDIS_DATABASES[environment][dataset_name]
        self._session_token = session_token

    def _authenticate(self):
        api_server = self._api_server or API_SERVER_URLS[self._environment]
        if self._auth:
            if not self._session_token:
                authenticator = KratosAuthenticator(
                    self._auth_url, self._username, self._password
                )
                #  TODO: Delete this once the prod cluster is updated
                if self._environment == "prod":
                    self._session_token = authenticator.authenticate_prod()
                else:
                    self._session_token = authenticator.authenticate()

            client = Client(api_server=api_server, api_token=self._session_token)
        else:
            client = Client(api_server=API_SERVER_URLS[self._environment])

        return client

    def _make_flow_config(self):
        return make_flow_config(
            image=self._image,
            storage_type=PrefectStorageMethods.S3,
            executor_type=PrefectExecutorMethods.LOCAL_DASK,
            redis_db=self._redis_db,
            environment=self._environment,
            schedule=self._schedule,
        )
