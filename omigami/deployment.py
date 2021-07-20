from datetime import timedelta
from typing import Optional

from prefect import Client
from typing_extensions import Literal

from omigami.authentication.authenticator import KratosAuthenticator
from omigami.config import (
    API_SERVER_URLS,
    MLFLOW_SERVER,
    PROJECT_NAME,
    REDIS_DATABASES,
    DATASET_IDS,
    IonModes,
)


class Deployer:
    def __init__(
        self,
        image: str,
        dataset_name: str,
        mlflow_server: str = MLFLOW_SERVER,
        environment: Literal["dev", "prod"] = "dev",
        overwrite: bool = False,
        overwrite_all: bool = True,
        auth: bool = False,
        auth_url: Optional[str] = None,
        ion_mode: IonModes = "positive",
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_server: Optional[str] = None,
        session_token: Optional[str] = None,
        project_name: str = PROJECT_NAME,
        deploy_model: bool = False,
        schedule: timedelta = None,
    ):
        self._image = image
        self._mlflow_server = mlflow_server
        self._auth = auth
        self._ion_mode = ion_mode
        self._deploy_model = deploy_model
        self._schedule = schedule
        self._overwrite = overwrite
        self._overwrite_all = overwrite_all

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
        self._project_name = project_name
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
                self._session_token = authenticator.authenticate()

            client = Client(api_server=api_server, api_token=self._session_token)
        else:
            client = Client(api_server=API_SERVER_URLS[self._environment])

        return client
