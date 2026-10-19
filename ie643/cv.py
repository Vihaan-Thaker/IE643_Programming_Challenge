"""Phase 0: the fixed validation protocol.

Every training row gets a group:
- "clean":    is_deformed == 0
- "far":      deformed rows in the separate far group found in eda_deformed.ipynb, Section 3.1
- "deformed": all other deformed rows

Labelled rows are split into 5 folds, stratified on class x group, and the assignment is saved to
folds/cv_folds_v1.json so that every experiment uses exactly the same folds.
"""
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .data import N_CLASSES, REPO_ROOT

GROUPS = ("clean", "deformed", "far")
FAR_PC1 = 15.0
N_FOLDS = 5
FOLD_SEED = 0
FOLDS_FILE = REPO_ROOT / "folds" / "cv_folds_v1.json"


def _pc1_projector(E, deformed):
    """Function giving PC1 of any rows, from a 2-component PCA of the standardized training features.

    PCA signs are arbitrary, so PC1 is oriented to point from the clean mean towards the deformed mean
    (the far group then sits at large positive PC1, as in the EDA notebook).
    """
    scaler = StandardScaler().fit(E)
    pca = PCA(n_components=2, random_state=0).fit(scaler.transform(E))
    pc1 = lambda X: pca.transform(scaler.transform(X))[:, 0]
    s = pc1(E)
    sign = 1.0 if s[deformed].mean() > s[~deformed].mean() else -1.0
    return lambda X: sign * pc1(X)


def assign_groups(E, deformed, X=None):
    """Group ("clean" / "deformed" / "far") of each training row.

    With X (e.g. test rows, which have no flag), returns only whether each row of X falls in the far group.
    """
    project = _pc1_projector(E, deformed)
    if X is None:
        far = deformed & (project(E) > FAR_PC1)
        return np.where(~deformed, "clean", np.where(far, "far", "deformed"))
    return project(X) > FAR_PC1          # for unflagged rows (test): only "in the far group or not"


def make_folds(y, groups, n_splits=N_FOLDS, seed=FOLD_SEED):
    """Fold index (0..n_splits-1) for labelled rows, -1 for unlabelled rows."""
    labelled = y >= 0
    strata = np.char.add(np.char.add(y[labelled].astype(str), "|"), groups[labelled].astype(str))
    fold = np.full(len(y), -1)
    idx = np.flatnonzero(labelled)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, val) in enumerate(skf.split(idx, strata)):
        fold[idx[val]] = k
    return fold


def save_folds(sample_id, groups, fold, path=FOLDS_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"sample_id": int(s), "group": str(g), "fold": int(f)} for s, g, f in zip(sample_id, groups, fold)]
    meta = {"description": "Fixed CV folds for labelled training rows (fold -1 = unlabelled). "
                           "Stratified on class x group; see ie643/cv.py.",
            "n_folds": N_FOLDS, "fold_seed": FOLD_SEED, "far_pc1_cut": FAR_PC1}
    path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=0))


def load_folds(sample_id, path=FOLDS_FILE):
    """Groups and folds aligned with the given sample_id order."""
    rows = pd.DataFrame(json.loads(path.read_text())["rows"]).set_index("sample_id").loc[sample_id]
    return rows["group"].to_numpy(), rows["fold"].to_numpy()


def evaluate(y_true, y_pred, groups):
    """Metrics reported for every experiment. Balanced accuracy is the main one (stand-in for the weighted metric)."""
    out = {"balanced accuracy": balanced_accuracy_score(y_true, y_pred),
           "accuracy": accuracy_score(y_true, y_pred)}
    for g in GROUPS:
        m = groups == g
        out[f"accuracy: {g}"] = accuracy_score(y_true[m], y_pred[m]) if m.any() else np.nan
    return out


def per_class_recall(y_true, y_pred):
    return pd.Series({c: np.mean(y_pred[y_true == c] == c) for c in range(N_CLASSES)}, name="recall")
