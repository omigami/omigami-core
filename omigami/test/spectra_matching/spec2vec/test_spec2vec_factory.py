from prefect import Flow

from omigami.config import MLFLOW_SERVER, OMIGAMI_ENV
from omigami.spectra_matching.spec2vec.factory import Spec2VecFlowFactory
from omigami.spectra_matching.tasks import CreateChunks


def test_build_training_flow():
    factory = Spec2VecFlowFactory()
    flow = factory.build_training_flow(
        flow_name="Robert DeFlow",
        image="image",
        iterations=5,
        window=10,
        intensity_weighting_power=0.4,
        allowed_missing_percentage=10,
        dataset_id="small",
        n_decimals=2,
        schedule=None,
        ion_mode="positive",
        source_uri="None",
        overwrite_all_spectra=False,
        overwrite_model=True,
        project_name="Raging Flow",
        deploy_model=False,
        chunk_size=int(1e8),
    )

    assert isinstance(flow, Flow)
    assert flow.name == "Robert DeFlow"
    assert len(flow.tasks) == 8
    chunk_task: CreateChunks = flow.get_tasks("CreateChunks")[0]
    assert chunk_task._chunk_size == int(1e8)


def test_build_model_deployment_flow():
    factory = Spec2VecFlowFactory()
    flow = factory.build_model_deployment_flow(
        flow_name="Calvin Flowers",
        image="star wars episode II wasn't so bad",
        intensity_weighting_power=0.4,
        allowed_missing_percentage=10,
        dataset_id="small",
        n_decimals=2,
        ion_mode="positive",
        project_name="Raging Flow",
    )

    assert isinstance(flow, Flow)
    assert flow.name == "Calvin Flowers"
    assert len(flow.tasks) == 5
    assert flow.run_config.env == {
        "REDIS_DB": "0",
        "REDIS_HOST": "localhost",
        "OMIGAMI_ENV": OMIGAMI_ENV,
        "MLFLOW_SERVER": MLFLOW_SERVER,
        "TZ": "UTC",
    }
