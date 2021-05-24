import pytest
from drfs import DRPath

from spec2vec_mlops.deployment import (
    deploy_training_flow,
    API_SERVER,
    SOURCE_URI_PARTIAL_GNPS,
    OUTPUT_DIR,
    MODEL_DIR,
    MLFLOW_SERVER,
    SOURCE_URI_COMPLETE_GNPS,
)


@pytest.mark.skip(
    reason="This test uses internet connection and deploys a test flow to prefect."
)
def test_deploy_training_flow(auth):
    flow_id = deploy_training_flow(
        image="drtools/prefect:spec2vec_mlops-SNAPSHOT.1f9bf5b",
        iterations=5,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        n_decimals=2,
        skip_if_exists=True,
        auth=True,
        auth_url=auth["auth_url"],
        username=auth["username"],
        password=auth["pwd"],
        api_server=API_SERVER["remote"],
        dataset="10k",
        source_uri=SOURCE_URI_COMPLETE_GNPS,
        output_dir=OUTPUT_DIR,
        project_name="spec2vec-mlops-10k",
        model_output_dir=str(DRPath(f"{MODEL_DIR}/tests")),
        mlflow_server=MLFLOW_SERVER,
        flow_name="training-flow",
        redis_db="1",
    )

    assert flow_id
