import os

from spec2vec_mlops.helper_classes.model_register import ModelRegister
from spec2vec_mlops.helper_classes.spec2vec_model import Model


def test_get_or_create_experiment(tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(f"file:/{path}")
    experiment_name = "experiment"
    # create
    created_experiment_id = model_register._get_or_create_experiment_id(
        experiment_name, path
    )
    # get
    experiment_id = model_register._get_or_create_experiment_id(experiment_name, path)
    assert created_experiment_id == experiment_id


def test_register_model(word2vec_model, tmpdir):
    path = f"{tmpdir}/mlflow/"
    model_register = ModelRegister(f"file:/{path}")
    model = Model(
        word2vec_model,
        n_decimals=1,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5.0,
    )
    run_id = model_register.register_model(
        model=model,
        params={
            "n_decimals_for_documents": model.n_decimals,
        },
        metrics={"alpha": model.model.alpha},
        experiment_name="experiment",
        path=path,
        code_to_save=["../../../spec2vec_mlops"],
    )
    assert run_id
    assert os.path.exists(f"{path}/model/python_model.pkl")
    assert os.path.exists(f"{path}/model/conda.yaml")
