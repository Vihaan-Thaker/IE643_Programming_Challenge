import sys, time, pickle
sys.path.insert(0, sys.argv[1]); PKL = sys.argv[2]
import numpy as np, pandas as pd
from ie643 import cv, data, mlp, distill

d = data.load()
groups, fold = cv.load_folds(d.train_id)
lab = d.labelled
CFG = mlp.Config(epochs=40, dropout=0.5, input_dropout=0.2, weight_decay=0.05, loss="logit_adjusted")
cols = ["balanced accuracy", "accuracy", "accuracy: clean", "accuracy: deformed", "accuracy: far"]
tlog = pickle.load(open(PKL, "rb"))["teachers"]["G"]
toof = pickle.load(open(PKL.replace(".pkl", "c.pkl"), "rb"))["teacher_oof"]
SCFG = distill.StudentConfig(lam=2.0, T=2.0, lam_lab=1.0, beta=1.0)
t = time.time()
base, _, _ = mlp.run_cv(d.E[lab], d.y[lab], groups[lab], fold[lab], CFG, seeds=(0, 1, 2))
stud, _ = distill.run_student_cv(d.E, d.y, groups, fold, tlog, CFG, SCFG, seeds=(0, 1, 2), G=d.G, teacher_oof=toof)
print(f"({time.time()-t:.0f}s)")
print("baseline:", base[cols].mean().round(3).to_dict(), "| bal std", round(base["balanced accuracy"].std(), 3))
print("student :", stud[cols].mean().round(3).to_dict(), "| bal std", round(stud["balanced accuracy"].std(), 3))
for metric in ["balanced accuracy", "accuracy"]:
    diff = stud[metric].to_numpy() - base[metric].to_numpy()
    print(f"paired {metric}: {diff.mean():+.3f} (sd {diff.std(ddof=1):.3f}, se {diff.std(ddof=1)/np.sqrt(len(diff)):.3f}), higher in {(diff > 0).sum()}/{len(diff)} runs")
