from omigami.spec2vec.factory import Spec2VecFlowFactory


def test_build_training_flow():
    factory = Spec2VecFlowFactory("dev")
    assert factory.build_training_flow()


def test_build_model_deployment_flow():
    factory = Spec2VecFlowFactory("dev")
    assert factory.build_model_deployment_flow()
