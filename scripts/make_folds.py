"""Phase 0: create folds/cv_folds_v1.json (run once from the repository root: python scripts/make_folds.py)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ie643 import cv, data  # noqa: E402

d = data.load()
groups = cv.assign_groups(d.E, d.deformed)
fold = cv.make_folds(d.y, groups)
cv.save_folds(d.train_id, groups, fold)

lab = d.labelled
print("groups, all training rows:", pd.Series(groups).value_counts().to_dict())
print("test rows in the far group:", int(cv.assign_groups(d.E, d.deformed, d.F).sum()))
print("\nlabelled rows per fold and group:")
print(pd.crosstab(fold[lab], groups[lab], margins=True).to_string())
print("\nlabelled rows per fold and class:")
print(pd.crosstab(fold[lab], d.y[lab]).to_string())
print("\nwrote", cv.FOLDS_FILE.relative_to(data.REPO_ROOT))
