from typing import Optional

from prefect import Client

from omigami.authentication.authenticator import KratosAuthenticator


class PrefectAuthenticator:
    def __init__(
        self,
        auth: bool = False,
        auth_url: Optional[str] = None,
        api_server: Optional[str] = None,
        session_token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self._auth = auth
        self._auth_url = auth_url
        self._api_server = api_server
        self._session_token = session_token
        self._username = username
        self._password = password

    def get_client(self) -> Client:
        if self._auth:
            if not self._session_token:
                authenticator = KratosAuthenticator(
                    self._auth_url, self._username, self._password
                )
                self._session_token = authenticator.authenticate()

            client = Client(api_server=self._api_server, api_token=self._session_token)
        else:
            client = Client(api_server=self._api_server)

        return client
