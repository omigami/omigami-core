import pytest
from prefect import Flow

from omigami.ms2deepscore.factory import MS2DeepScoreFlowFactory


def test_build_ms2deep_training_flow():
    factory = MS2DeepScoreFlowFactory("dev")
    ms2deep_training_flow = factory.build_training_flow(
        flow_name="MS2DeepScore Training Flow",
        image="image",
        dataset_name="small",
        schedule=None,
        ion_mode="positive",
        source_uri="URI",
        overwrite_all_spectra=False,
        overwrite_model=True,
        project_name="Raging Flow",
        deploy_model=False,
        spectrum_ids_chunk_size=10000,
        fingerprint_n_bits=2048,
        scores_decimals=5,
        spectrum_binner_n_bins=10000,
    )

    assert isinstance(ms2deep_training_flow, Flow)
    assert ms2deep_training_flow.name == "MS2DeepScore Training Flow"
    assert len(ms2deep_training_flow.tasks) == 9
    tanimoto_task = ms2deep_training_flow.get_tasks("CalculateTanimotoScore")[0]
    assert tanimoto_task._decimals == 5


@pytest.mark.xfail(reason="Not implemented atm.")  # TODO
def test_build_model_deployment_flow():
    factory = MS2DeepScoreFlowFactory("dev")
    assert factory.build_model_deployment_flow()
