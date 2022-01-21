from pathlib import Path

import mlflow
from omigami.spectra_matching.spec2vec.factory import Spec2VecFlowFactory

storage_root = Path.cwd() / "results"


if __name__ == "__main__":
    factory = Spec2VecFlowFactory(
        storage_root=storage_root,
        model_registry_uri="sqlite:///mlflow.sqlite",
        mlflow_output_directory=str(storage_root / "spec2vec/models"),
    )
    flow = factory.build_training_flow(
        flow_name="spec2vec",
        iterations=5,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5.0,
        n_decimals=2,
        dataset_id="complete",
        ion_mode="positive",
        project_name="spec2vec-positive",
    )
    flow_run = flow.run()

    register_task = flow.get_tasks("RegisterModel")[0]
    run_id = flow_run.result[register_task].result

    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    model_uri = f"{artifact_uri}/model/python_model.pkl"
    print(f"Spec2Vec model is available at: {model_uri}")
