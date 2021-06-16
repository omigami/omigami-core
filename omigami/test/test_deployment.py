import pytest
from drfs import DRPath

from omigami.config import (
    SOURCE_URI_COMPLETE_GNPS,
    S3_BUCKET,
    MODEL_DIR,
    MLFLOW_SERVER,
    config,
    SOURCE_URI_PARTIAL_GNPS,
)
from omigami.deployment import (
    deploy_training_flow,
)


@pytest.mark.skip(
    reason="This test uses internet connection and deploys a test flow to prefect."
)
def test_deploy_training_flow():
    login_config = config["login"]["dev"].get(dict)
    login_config.pop("token")
    flow_id = deploy_training_flow(
        image="drtools/prefect:omigami-SNAPSHOT.bc19d2b",
        iterations=3,
        window=300,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        n_decimals=2,
        skip_if_exists=True,
        chunk_size=int(1e8),
        environment="dev",
        ion_mode="negative",
        dataset_name="complete",
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        output_dir=S3_BUCKET["dev"],
        project_name="spec2vec-test",
        model_output_dir=MODEL_DIR["dev"],
        mlflow_server=MLFLOW_SERVER,
        flow_name="training-flow/chunk-by-ion-mode",
        deploy_model=True,
        auth=True,
        **login_config,
    )

    assert flow_id


@pytest.mark.skip(reason="This test uses internet connection.")
def test_dataset_wrong_dataset_name():
    with pytest.raises(ValueError):
        flow_id = deploy_training_flow(
            dataset_name="NOT-A-DATASET",
            image="drtools/prefect:omigami-SNAPSHOT.f06b4f9",
            iterations=5,
            window=500,
            intensity_weighting_power=0.5,
            allowed_missing_percentage=5,
            n_decimals=2,
            skip_if_exists=True,
            source_uri=SOURCE_URI_COMPLETE_GNPS,
            environment="dev",
            output_dir=S3_BUCKET,
            project_name="spec2vec-mlops-10k",
            model_output_dir=str(DRPath(f"{MODEL_DIR}/tests")),
            mlflow_server=MLFLOW_SERVER,
            flow_name="training-flow",
            auth=True,
            **config["login"]["dev"].get(dict),
        )
