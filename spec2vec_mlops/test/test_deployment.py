import pytest
from drfs import DRPath

from spec2vec_mlops.deployment import (
    deploy_training_flow,
    API_SERVER,
    DATASET_NAME,
    SOURCE_URI_PARTIAL_GNPS,
    OUTPUT_DIR,
    MODEL_DIR,
    MLFLOW_SERVER,
)


@pytest.mark.skip(
    reason="This test uses internet connection and deploys a test flow to prefect."
)
def test_deploy_training_flow():
    flow_id = deploy_training_flow(
        image="drtools/prefect:spec2vec_mlops-SNAPSHOT.bff888c",
        iterations=5,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        n_decimals=2,
        skip_if_exists=True,
        auth=False,
        auth_url=None,
        username=None,
        password=None,
        api_server=API_SERVER["remote"],
        dataset_name=DATASET_NAME,
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        output_dir=OUTPUT_DIR,
        project_name="spec2vec-mlops-test-flow",
        model_output_dir=str(DRPath(f"{MODEL_DIR}/tests")),
        mlflow_server=MLFLOW_SERVER,
    )

    assert flow_id
