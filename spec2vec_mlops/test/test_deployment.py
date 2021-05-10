import pytest
from drfs import DRPath

from spec2vec_mlops.deployment import (
    deploy_training_flow,
    API_SERVER,
    DATASET_NAME,
    SOURCE_URI_PARTIAL_GNPS,
    OUTPUT_DIR,
    MODEL_DIR,
    SELDON_DEPLOYMENT_PATH,
    MLFLOW_SERVER,
)


# @pytest.mark.skip(reson="This test uses internet connection and deploys to prefect.")
def test_deploy_training_flow():
    flow_id = deploy_training_flow(
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
        project_name="spec2vec-mlops-test-project",
        model_output_dir=DRPath(f"{MODEL_DIR}/tests"),
        seldon_deployment_path=SELDON_DEPLOYMENT_PATH,
        mlflow_server=MLFLOW_SERVER,
        image="drtools/prefect:spec2vec_mlops-SNAPSHOT.976bf73",
    )

    assert flow_id
