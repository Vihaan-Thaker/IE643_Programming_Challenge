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
ref = pickle.load(open(PKL.replace(".pkl", "c.pkl"), "rb"))["student (soft T=2, lam=2, threshold=0, lam_lab=1, G-head beta=1)"][0]
grid = {
    "epochs 60": (replace(CFG, epochs=60), S),
    "epochs 25": (replace(CFG, epochs=25), S),
    "wide 1024-512": (replace(CFG, hidden=(1024, 512)), S),
    "dropout .3 / input .1": (replace(CFG, dropout=0.3, input_dropout=0.1), S),
    "beta 3": (CFG, replace(S, beta=3.0)),
    "lam_lab 3": (CFG, replace(S, lam_lab=3.0)),
    "T 4": (CFG, replace(S, T=4.0)),
    "tau 1.5": (replace(CFG, tau=1.5), S),
}
print("reference (all three channels, seed 0):", ref[cols].mean().round(3).to_dict(), flush=True)
out = {}
for name, (cfg, scfg) in grid.items():
    t = time.time()
    m, oof = distill.run_student_cv(d.E, d.y, groups, fold, tlog, cfg, scfg, seeds=(0,), G=d.G, teacher_oof=toof)
    diff = m["balanced accuracy"].to_numpy() - ref["balanced accuracy"].to_numpy()
    print(f"{name} ({time.time()-t:.0f}s):", m[cols].mean().round(3).to_dict(), "| paired vs ref %+.3f (%d/5)" % (diff.mean(), (diff > 0).sum()), flush=True)
    out[name] = (m, oof)
pickle.dump(out, open(PKL.replace(".pkl", "_sweep.pkl"), "wb"))
