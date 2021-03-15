from pathlib import Path

from prefect import Flow

from spec2vec_mlops import config
from spec2vec_mlops.tasks.load_data import load_data_task

# variable definitions region
TEST_URI_PATH = str(Path(__file__).parents[1] / "test" / "assets" / "SMALL_GNPS.json")
TEST_URI = f"file://{TEST_URI_PATH}"
SOURCE_URI = config["gnps_json"]["uri"].get(str)
URI = TEST_URI  # SOURCE_URI
# TODO: change it to source URI

with Flow("spec2vec-training-flow") as training_flow:
    raw = load_data_task(URI)
    # cleaned = clean_data_task(raw)
    # saved = save_data_to_feast_task(cleaned)
    # documents = convert_data_to_documents_task(saved)
    # encoded = encode_training_data_task(documents)
    # trained = train_model_task(documents)


training_flow_state = training_flow.run()
