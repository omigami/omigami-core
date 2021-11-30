import pytest
from prefect import Flow

from omigami.spectra_matching.spec2vec.factory import Spec2VecFlowFactory
from omigami.spectra_matching.tasks import CreateChunks


def test_build_training_flow():
    factory = Spec2VecFlowFactory("dev")
    flow = factory.build_spec2vec_flow(
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
    assert len(flow.tasks) == 7
    chunk_task: CreateChunks = flow.get_tasks("CreateChunks")[0]
    assert chunk_task._chunk_size == int(1e8)


@pytest.mark.xfail(reason="Not implemented atm.")  # TODO
def test_build_model_deployment_flow():
    factory = Spec2VecFlowFactory("dev")
    assert factory.build_model_deployment_flow()
