"""Loading the competition data into numpy arrays."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "ie-643-2026-programming-challenge"
N_CLASSES = 11


@dataclass
class Data:
    train_id: np.ndarray   # (2076,) sample_id of each training row
    E: np.ndarray          # (2076, 768) main training features
    G: np.ndarray          # (2076, 384) supplementary features, training rows only
    y: np.ndarray          # (2076,) class label, -1 = unlabelled
    deformed: np.ndarray   # (2076,) bool, is_deformed == 1
    test_id: np.ndarray    # (490,) sample_id of each test row
    F: np.ndarray          # (490, 768) test features

    @property
    def labelled(self):
        return self.y >= 0


def load(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "Train_Data.csv")
    test = pd.read_csv(data_dir / "Test_Data.csv")
    supp = pd.read_csv(data_dir / "Supplementry_Data.csv")

    g_cols = [f"g{i}" for i in range(384)]
    supp = supp[~supp[g_cols].isna().all(axis=1)].reset_index(drop=True)   # file ends with 2 empty rows
    assert (supp["sample_id"].to_numpy() == train["sample_id"].to_numpy()).all()

    return Data(
        train_id=train["sample_id"].to_numpy(),
        E=train[[f"e{i}" for i in range(768)]].to_numpy(np.float32),
        G=supp[g_cols].to_numpy(np.float32),
        y=train["Class_Label"].to_numpy(),
        deformed=train["is_deformed"].to_numpy() == 1,
        test_id=test["sample_id"].to_numpy(),
        F=test[[f"f{i}" for i in range(768)]].to_numpy(np.float32),
    )
