import logging

import requests

logger = logging.getLogger(__name__)


class UnauthorizedError(Exception):
    pass


class BadRequestError(Exception):
    pass


class Authenticator:
    def __init__(self, username: str, password: str):
        self._username = username
        self._password = password

    def authenticate(self):
        pass


class KratosAuthenticator(Authenticator):
    def __init__(self, host: str, username: str, password: str):
        """Implements an Username/Password authenticator for ORY Kratos

        Parameters
        ----------
        host: str
            ORY Kratos Public URL:
                e.g.: https://mlops.datarevenue.com/.ory/kratos/public/
        username: str
            Authentication username / e-mail
        password: str
            Authentication password
        """
        if host[-1] != "/":
            host = f"{host}/"
        self.url = host
        super(KratosAuthenticator, self).__init__(username, password)

    def authenticate(self):
        action_url = self._get_login_flow_url()
        session_token = self._get_session_token(action_url)
        return session_token

    def _get_login_flow_url(self):
        """Sends a request to Krato's Public endpoint to get the Login Flow URL + UID"""
        r = requests.get(f"{self.url}self-service/login/api")
        if r.status_code != 200:
            logger.error("Authentication Failed")
            logger.error(r.text)
            r.raise_for_status()
        json = r.json()
        try:
            action_url = json["ui"]["action"]
        except KeyError as e:
            logger.error(
                "Unable to get ORY's Login Flow. Please check the configuration"
            )
            raise e

        return action_url

    def _get_session_token(self, action_url):
        """Authenticate the user using the provided credentials"""
        data = {
            "method": "password",
            "password_identifier": self._username,
            "password": self._password,
        }
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        r = requests.post(action_url, json=data, headers=headers)
        if r.status_code == 401:
            raise UnauthorizedError("Please review your auth credentials.")
        elif r.status_code == 404:
            raise BadRequestError(
                "Something went wrong while trying to get session token."
            )
        json = r.json()
        try:
            session_token = json["session_token"]
        except KeyError as e:
            logger.error("Error while trying to fetch Session Token")
            raise e
        return session_token

    #  TODO: Delete this once the prod cluster is updated
    def authenticate_prod(self):
        flow = requests.get(
            "https://omigami.datarevenue.com/.ory/kratos/public/self-service/login/api"
        )
        action_url = flow.json()["methods"]["password"]["config"]["action"]
        auth = requests.post(
            action_url, json={"identifier": self._username, "password": self._password}
        )
        token = auth.json()["session_token"]
        return token
