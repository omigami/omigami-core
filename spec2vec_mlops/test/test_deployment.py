import pytest

from drfs import DRPath

from spec2vec_mlops.deployment import (
    deploy_training_flow,
)
from spec2vec_mlops import (
    SOURCE_URI_PARTIAL_GNPS,
    API_SERVER,
    OUTPUT_DIR,
    MODEL_DIR,
    MLFLOW_SERVER,
)


def test_deploy_training_flow():
    flow_id = deploy_training_flow(
        image="drtools/prefect:spec2vec_mlops-SNAPSHOT.f06b4f9",
        iterations=5,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        n_decimals=2,
        skip_if_exists=True,
        auth=True,
        auth_url="https://mlops.datarevenue.com/.ory/kratos/public/",
        username="ofiehn@ucdavis.edu",
        password="PWspec2vecbeta",
        api_server=API_SERVER["remote"],
        dataset_name=None,
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        output_dir=OUTPUT_DIR,
        project_name="spec2vec-mlops-test-flow",
        model_output_dir=str(DRPath(f"{MODEL_DIR}/tests")),
        mlflow_server=MLFLOW_SERVER,
    )

    assert flow_id
