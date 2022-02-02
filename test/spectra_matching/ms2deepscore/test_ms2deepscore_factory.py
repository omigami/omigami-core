from prefect import Flow

from omigami.config import STORAGE_ROOT, MLFLOW_SERVER, REDIS_HOST, OMIGAMI_ENV
from omigami.spectra_matching.ms2deepscore.factory import MS2DeepScoreFlowFactory
from omigami.spectra_matching.storage import REDIS_DB


def test_build_ms2deep_training_flow():
    factory = MS2DeepScoreFlowFactory()
    ms2deep_training_flow = factory.build_training_flow(
        flow_name="MS2DeepScore Training Flow",
        image="image",
        dataset_id="small",
        schedule=None,
        ion_mode="positive",
        project_name="Raging Flow",
        spectrum_ids_chunk_size=10000,
        fingerprint_n_bits=2048,
        scores_decimals=5,
        spectrum_binner_n_bins=10000,
    )

    assert isinstance(ms2deep_training_flow, Flow)
    assert ms2deep_training_flow.name == "MS2DeepScore Training Flow"
    assert len(ms2deep_training_flow.tasks) == 7
    tanimoto_task = ms2deep_training_flow.get_tasks("CalculateTanimotoScore")[0]
    assert tanimoto_task._decimals == 5
    assert ms2deep_training_flow.storage.directory == str(STORAGE_ROOT)
    assert ms2deep_training_flow.run_config.env == {
        "REDIS_DB": REDIS_DB,
        "REDIS_HOST": REDIS_HOST,
        "OMIGAMI_ENV": OMIGAMI_ENV,
        "MLFLOW_SERVER": MLFLOW_SERVER,
        "TZ": "UTC",
    }


def test_build_model_deployment_flow():
    factory = MS2DeepScoreFlowFactory()
    model_deployment_flow = factory.build_model_deployment_flow(
        flow_name="Model Destroyment Flow",
        image="star wars episode III had nice lightsaber fights",
        dataset_id="small",
        ion_mode="positive",
        project_name="Raging Flow",
    )

    assert isinstance(model_deployment_flow, Flow)
    assert model_deployment_flow.name == "Model Destroyment Flow"
    assert len(model_deployment_flow.tasks) == 8
    assert model_deployment_flow.run_config.env == {
        "REDIS_DB": REDIS_DB,
        "REDIS_HOST": REDIS_HOST,
        "OMIGAMI_ENV": OMIGAMI_ENV,
        "MLFLOW_SERVER": MLFLOW_SERVER,
        "TZ": "UTC",
    }
