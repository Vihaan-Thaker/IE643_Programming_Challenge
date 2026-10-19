"""How much of G can be predicted from E, and how well can one classify from predicted G?"""
import sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd, torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from ie643 import cv, data, mlp
torch.set_num_threads(3)          # share the CPU with the main job

d = data.load()
groups, fold = cv.load_folds(d.train_id)
lab = fold >= 0
Ez = StandardScaler().fit_transform(d.E)
Gz = StandardScaler().fit_transform(d.G)

# 1. Out-of-fold prediction of G from E over all 2076 rows (no labels involved)
for alpha in (10.0, 100.0, 1000.0):
    Ghat = np.zeros_like(Gz)
    for tr, va in KFold(5, shuffle=True, random_state=0).split(Ez):
        Ghat[va] = Ridge(alpha=alpha).fit(Ez[tr], Gz[tr]).predict(Ez[va])
    r2 = 1 - ((Gz - Ghat) ** 2).sum(0) / (Gz ** 2).sum(0)
    print(f"ridge alpha={alpha:g}: mean R2 over the 384 G dims {r2.mean():.3f} (median {np.median(r2):.3f}, "
          f"dims with R2>0.5: {(r2 > 0.5).sum()})", flush=True)
    for g in cv.GROUPS:
        m = groups == g
        r2g = 1 - ((Gz[m] - Ghat[m]) ** 2).sum(0) / ((Gz[m] - Gz[m].mean(0)) ** 2).sum(0)
        print(f"   within {g}: mean R2 {r2g.mean():.3f}", flush=True)
    if alpha == 100.0:
        Ghat_best = Ghat.copy()

# 2. Classify from predicted G (labelled rows, fixed folds)
cfg = mlp.Config(epochs=40, dropout=0.5, input_dropout=0.2, weight_decay=0.05, loss="logit_adjusted")
rows = []
t = time.time()
for k in range(cv.N_FOLDS):
    tr, va = lab & (fold != k), fold == k
    teacher, _ = mlp.fit(Gz[tr], d.y[tr], cfg, seed=0)                     # trained on true G
    rows.append({"route": "teacher(true G) applied to predicted G", **cv.evaluate(d.y[va], mlp.predict_proba(teacher, Ghat_best[va]).argmax(1), groups[va])})
    clf, _ = mlp.fit(Ghat_best[tr], d.y[tr], cfg, seed=0)                  # trained on predicted G
    rows.append({"route": "MLP trained on predicted G", **cv.evaluate(d.y[va], mlp.predict_proba(clf, Ghat_best[va]).argmax(1), groups[va])})
    rows.append({"route": "teacher on true G (reference)", **cv.evaluate(d.y[va], mlp.predict_proba(teacher, Gz[va]).argmax(1), groups[va])})
print(f"({time.time()-t:.0f}s)")
cols = ["balanced accuracy", "accuracy", "accuracy: clean", "accuracy: deformed", "accuracy: far"]
print(pd.DataFrame(rows).groupby("route", sort=False)[cols].mean().round(3).to_string())
