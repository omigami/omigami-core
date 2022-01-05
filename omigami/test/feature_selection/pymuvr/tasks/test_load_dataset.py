import dask.dataframe as dd
from pathlib import Path

from omigami.feature_selection.pymuvr.tasks.load_dataset import LoadDataset

ASSETS_DIR = Path(__file__).parents[3] / "assets" / "feature_selection" / "pymuvr"


def test_load_dataset():
    dataset_path = str(ASSETS_DIR / "training_dataset.csv")
    res = LoadDataset().run(dataset_path)

    assert isinstance(res, dd.DataFrame)
    assert len(res) > 0
