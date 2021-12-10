import pytest

from omigami.authentication.prefect_factory import prefect_client_factory


@pytest.fixture()
def prefect_service():
    prefect_client = prefect_client_factory.get()

    if not prefect_client.active_tenant_id:
        prefect_client.create_tenant("default")

    return prefect_client
