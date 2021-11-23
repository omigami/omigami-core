from unittest.mock import Mock

import pytest
from prefect import Client

import omigami.authentication.prefect_factory
from omigami.authentication.authenticator import KratosAuthenticator
from omigami.authentication.prefect_factory import PrefectClientFactory
from omigami.config import API_SERVER_URLS, config


@pytest.fixture()
def mock_prefect_client(monkeypatch):
    MockPrefectClient = Mock()
    monkeypatch.setattr(
        omigami.authentication.prefect_factory, "Client", MockPrefectClient
    )

    return MockPrefectClient()


def test_mock_prefect_get_client_with_token(mock_prefect_client):
    auth = Mock(spec=KratosAuthenticator)

    factory = PrefectClientFactory(
        "auth-url", "api-server", session_token="token", authenticator=auth
    )

    client = factory.get_client()

    assert client == mock_prefect_client
    auth.authenticate.assert_not_called()


def test_mock_prefect_get_client_without_token(mock_prefect_client):
    auth = Mock(spec=KratosAuthenticator)

    factory_no_token = PrefectClientFactory(
        "auth-url", "api-server", authenticator=auth
    )

    client = factory_no_token.get_client()

    assert client == mock_prefect_client
    auth.authenticate.assert_called_once()


@pytest.mark.skip(
    "Requires local prefect server and mlflow. Make sure they are running to run this"
    "test. To run them, check README instructions."
)
def test_prefect_get_client():
    api_server = API_SERVER_URLS["local"]
    login_config = config["login"]["local"].get(dict)

    factory = PrefectClientFactory(api_server=api_server, **login_config)

    client = factory.get_client()

    assert isinstance(client, Client)
    assert client.api_server == api_server
    assert client.active_tenant_id is not None
