from spec2vec_mlops.flows.training_flow import training_flow_state


def test_spec2vec_training_flow():
    assert training_flow_state.is_successful()
