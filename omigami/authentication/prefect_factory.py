from typing import Optional

from prefect import Client

from omigami.authentication.authenticator import KratosAuthenticator, Authenticator


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
        if not self._session_token:
            self._session_token = self._authenticator.authenticate()

        client = Client(api_server=self._api_server, api_token=self._session_token)

        return client
