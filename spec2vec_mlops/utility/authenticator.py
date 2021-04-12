import logging

import requests

logger = logging.getLogger(__name__)


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
        self.url = host
        super(KratosAuthenticator, self).__init__(username, password)

    def authenticate(self):
        action_url = self._get_login_flow_url()
        session_token = self._get_session_token(action_url)
        return session_token

    def _get_login_flow_url(self):
        """Sends a request to Krato's Public endpoint to get the Login Flow URL + UID"""
        r = requests.get(self.url + "self-service/login/api")
        json = r.json()
        try:
            action_url = json["methods"]["password"]["config"]["action"]
        except KeyError as e:
            logger.error(
                "Unable to get ORY's Login Flow. Please check the configuration"
            )
            raise e

        return action_url

    def _get_session_token(self, action_url):
        """Authenticate the user using the provided credentials"""
        data = {"identifier": self._username, "password": self._password}
        headers = {"Content-type": "application/json"}
        r = requests.post(action_url, data=data, headers=headers)
        json = r.json()
        try:
            session_token = json["session_token"]
        except KeyError as e:
            logger.error("Error while trying to fetch Session Token")
            raise e
        return session_token
