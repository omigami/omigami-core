import pytest

from drfs import DRPath

from spec2vec_mlops import ENV
from spec2vec_mlops.deployment import (
    deploy_training_flow,
)

from spec2vec_mlops.config import (
    SOURCE_URI_COMPLETE_GNPS,
    S3_BUCKET,
    MODEL_DIR,
    MLFLOW_SERVER,
)


@pytest.mark.skip(
    reason="This test uses internet connection and deploys a test flow to prefect."
)
def test_deploy_training_flow():
    flow_id = deploy_training_flow(
        image="drtools/prefect:spec2vec_mlops-SNAPSHOT.1f9bf5b",
        iterations=5,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        n_decimals=2,
        skip_if_exists=True,
        auth=True,
        auth_url=ENV["auth_url"].get(),
        username=ENV["username"].get(),
        password=ENV["pwd"].get(),
        api_server=API_SERVER["remote"],
        auth_url="https://mlops.datarevenue.com/.ory/kratos/public/",
        username="ofiehn@ucdavis.edu",
        password="PWspec2vecbeta",
        environment="dev",
        dataset_name="10k",
        source_uri=SOURCE_URI_COMPLETE_GNPS,
        output_dir=S3_BUCKET,
        project_name="spec2vec-mlops-10k",
        model_output_dir=str(DRPath(f"{MODEL_DIR}/tests")),
        mlflow_server=MLFLOW_SERVER,
        flow_name="training-flow",
    )

    assert flow_id


@pytest.mark.skip(reason="This test uses internet connection.")
def test_dataset_wrong_dataset_name():
    with pytest.raises(ValueError):
        flow_id = deploy_training_flow(
            dataset_name="NOT-A-DATASET",
            image="drtools/prefect:spec2vec_mlops-SNAPSHOT.f06b4f9",
            iterations=5,
            window=500,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5,
            n_decimals=2,
            skip_if_exists=True,
            auth=True,
            auth_url=ENV["auth_url"].get(),
            username=ENV["username"].get(),
            password=ENV["pwd"].get(),
            api_server=API_SERVER["remote"],
            source_uri=SOURCE_URI_COMPLETE_GNPS,
            output_dir=S3_BUCKET,
            project_name="spec2vec-mlops-10k",
            model_output_dir=str(DRPath(f"{MODEL_DIR}/tests")),
            mlflow_server=MLFLOW_SERVER,
            flow_name="training-flow",
        )
