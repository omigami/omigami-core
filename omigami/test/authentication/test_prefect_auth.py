from omigami.authentication.prefect_auth import PrefectAuthenticator


def test_prefect_get_client():
    authenticator = PrefectAuthenticator()
    assert authenticator.get_client()
