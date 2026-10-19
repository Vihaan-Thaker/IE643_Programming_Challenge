import sys, time, pickle
sys.path.insert(0, sys.argv[1]); PKL = sys.argv[2]
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from ie643 import cv, data, mlp, distill

d = data.load()
groups, fold = cv.load_folds(d.train_id)
lab = d.labelled
CFG = mlp.Config(epochs=40, dropout=0.5, input_dropout=0.2, weight_decay=0.05, loss="logit_adjusted")
cols = ["balanced accuracy", "accuracy", "accuracy: clean", "accuracy: deformed", "accuracy: far"]
saved = pickle.load(open(PKL, "rb"))
tlog = saved["teachers"]["G"]
log_prior = np.log(np.bincount(d.y[lab], minlength=11) / lab.sum())

def posthoc(oof_lab, y_lab):
    # extra prior correction at prediction time: argmax(log p - t * log prior)
    return {f"+{t:g}": round(balanced_accuracy_score(y_lab, (np.log(oof_lab + 1e-12) - t * log_prior).argmax(1)), 3)
            for t in (0, 0.25, 0.5, 0.75, 1.0)}

t = time.time()
base, _, base_oof = mlp.run_cv(d.E[lab], d.y[lab], groups[lab], fold[lab], CFG, seeds=(0,))
print(f"baseline ({time.time()-t:.0f}s):", base[cols].mean().round(3).to_dict(), "| pooled OOF bal-acc with extra tau:", posthoc(base_oof, d.y[lab]), flush=True)
out = {"baseline": (base, base_oof)}
for scfg in [distill.StudentConfig(lam=1.0, T=1.0), distill.StudentConfig(lam=1.0, T=2.0), distill.StudentConfig(lam=2.0, T=2.0)]:
    t = time.time()
    m, oof = distill.run_student_cv(d.E, d.y, groups, fold, tlog, CFG, scfg, seeds=(0,))
    diff = m["balanced accuracy"].to_numpy() - base["balanced accuracy"].to_numpy()
    print(f"{scfg.name()} ({time.time()-t:.0f}s):", m[cols].mean().round(3).to_dict(),
          "| paired %+.3f (%d/5)" % (diff.mean(), (diff > 0).sum()), "| pooled OOF with extra tau:", posthoc(oof[lab], d.y[lab]), flush=True)
    out[scfg.name()] = (m, oof)
pickle.dump(out, open(PKL.replace(".pkl", "b.pkl"), "wb"))
