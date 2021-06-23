import os
import pytest
from typing import Optional
from omigami.config import SELDON_PARAMS
from omigami.seldon.seldon_deployment import SeldonDeployment

pytestmark = pytest.mark.skipif(
    os.getenv("SELDON_DEPLOYMENT_TEST", True),
    reason="It can only run if kubernetes cluster is available. "
    "Needs to proper set the kubecontext.",
)


@pytest.fixture
def kubeconfig() -> str:
    return "~/.kube/config"


@pytest.fixture
def kubecontext() -> Optional[str]:
    return "omigami-dev"


def test_seldon_deployment_init(kube, kubecontext):
    sd = SeldonDeployment(context=kubecontext)

    assert sd.client is not None
    assert isinstance(sd.deployment_spec, dict)


def test_list_seldon_deployments_empty(kube, kubecontext):
    seldon_config = SELDON_PARAMS
    seldon_config["namespace"] = kube.namespace

    sd = SeldonDeployment(context=kubecontext, seldon_config=seldon_config)
    deployments = sd.list_deployments()
    assert not deployments


def test_deploy_model(kube, kubecontext):
    seldon_config = SELDON_PARAMS
    seldon_config["namespace"] = kube.namespace

    sd = SeldonDeployment(context=kubecontext, seldon_config=seldon_config)
    res = sd.deploy_model(model_name="spec2vec-test", model_uri="s3://", redis_db="2")

    assert "spec2vec-test" in sd.list_deployments()


def test_config_deployment_spec(kube, kubecontext):
    seldon_config = SELDON_PARAMS
    seldon_config["namespace"] = kube.namespace
    sd = SeldonDeployment(context=kubecontext, seldon_config=seldon_config)

    model_name = "spec2vec-neutral"

    deployment = sd.config_deployment_spec(model_name, "model_uri", "2")

    assert deployment["spec"]["predictors"][0]["graph"]["modelUri"] == "model_uri"
    assert (
        deployment["spec"]["predictors"][0]["componentSpecs"][0]["spec"]["containers"][
            0
        ]["env"][-1]["value"]
        == "2"
    )
    assert deployment["metadata"]["name"] == model_name
    assert deployment["spec"]["name"] == model_name
    assert deployment["spec"]["predictors"][0]["graph"]["name"] == model_name
    assert (
        deployment["spec"]["predictors"][0]["componentSpecs"][0]["spec"]["containers"][
            0
        ]["name"]
        == model_name
    )
