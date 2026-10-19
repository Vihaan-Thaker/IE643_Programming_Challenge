import sys, time, pickle
from dataclasses import replace
sys.path.insert(0, sys.argv[1]); PKL = sys.argv[2]
import numpy as np, pandas as pd
from ie643 import cv, data, mlp, distill

d = data.load()
groups, fold = cv.load_folds(d.train_id)
CFG = mlp.Config(epochs=40, dropout=0.5, input_dropout=0.2, weight_decay=0.05, loss="logit_adjusted")
S = distill.StudentConfig(lam=2.0, T=2.0, lam_lab=1.0, beta=1.0)
cols = ["balanced accuracy", "accuracy", "accuracy: clean", "accuracy: deformed", "accuracy: far"]
tlog = pickle.load(open(PKL, "rb"))["teachers"]["G"]
toof = pickle.load(open(PKL.replace(".pkl", "c.pkl"), "rb"))["teacher_oof"]
grid = {"current (beta 1)": (CFG, S), "beta 3": (CFG, replace(S, beta=3.0)),
        "tau 1.5": (replace(CFG, tau=1.5), S), "beta 3 + tau 1.5": (replace(CFG, tau=1.5), replace(S, beta=3.0))}
res = {}
for name, (cfg, scfg) in grid.items():
    t = time.time()
    m, oof = distill.run_student_cv(d.E, d.y, groups, fold, tlog, cfg, scfg, seeds=(0, 1, 2), G=d.G, teacher_oof=toof)
    res[name] = (m, oof)
    print(f"{name} ({time.time()-t:.0f}s):", m[cols].mean().round(3).to_dict(), flush=True)
ref = res["current (beta 1)"][0]
for name in list(grid)[1:]:
    for metric in ["balanced accuracy", "accuracy", "accuracy: far"]:
        diff = res[name][0][metric].to_numpy() - ref[metric].to_numpy()
        print(f"{name} vs current, {metric}: {diff.mean():+.3f} (se {diff.std(ddof=1)/np.sqrt(len(diff)):.3f}, {(diff > 0).sum()}/15 higher)")
pickle.dump(res, open(PKL.replace(".pkl", "_confirm_tuned.pkl"), "wb"))
