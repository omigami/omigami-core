from typing import Optional

from prefect import Client

from omigami.authentication.authenticator import KratosAuthenticator, Authenticator
from omigami.config import API_SERVER_URLS, OMIGAMI_ENV, get_login_config


class PrefectClientFactory:
    def __init__(
        self,
        auth_url: Optional[str],
        api_server: Optional[str],
        session_token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        authenticator: Authenticator = None,
    ):
        self._api_server = api_server
        self._session_token = session_token
        self._authenticator = authenticator or KratosAuthenticator(
            auth_url, username, password
        )

    def get_client(self) -> Client:
        """Instantiates a Prefect Client using provided credentials"""
        if self._session_token is None:
            self._session_token = self._authenticator.authenticate()

        client = Client(api_server=self._api_server, api_token=self._session_token)

        return client


def get_prefect_client() -> Client:
    api_server = API_SERVER_URLS[OMIGAMI_ENV]
    login_config = get_login_config()
    prefect_factory = PrefectClientFactory(api_server=api_server, **login_config)
    prefect_client = prefect_factory.get_client()
    return prefect_client
