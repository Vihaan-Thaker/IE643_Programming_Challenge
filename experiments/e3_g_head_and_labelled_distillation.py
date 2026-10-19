import sys, time, pickle
sys.path.insert(0, sys.argv[1]); PKL = sys.argv[2]
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from ie643 import cv, data, mlp, distill

if len(sys.argv) > 3 and sys.argv[3] == "smoke":
    cfg = mlp.Config(epochs=1, loss="logit_adjusted"); rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20)).astype("float32"); G = rng.normal(size=(60, 8)).astype("float32")
    y = np.r_[np.arange(11), rng.integers(0, 11, 29), -np.ones(20, int)]
    fold = np.r_[np.arange(40) % 5, -np.ones(20, int)]; groups = np.array(["clean"] * 60)
    _, tl = distill.teacher_cv(G, y, groups, fold, cfg); to = distill.teacher_nested(G, y, fold, cfg)
    m, _ = distill.run_student_cv(X, y, groups, fold, tl, cfg, distill.StudentConfig(lam=1, T=2, lam_lab=1, beta=0.5, threshold=0.2), G=G, teacher_oof=to)
    print("smoke test ok", m.shape); sys.exit()

d = data.load()
groups, fold = cv.load_folds(d.train_id)
lab = d.labelled
CFG = mlp.Config(epochs=40, dropout=0.5, input_dropout=0.2, weight_decay=0.05, loss="logit_adjusted")
cols = ["balanced accuracy", "accuracy", "accuracy: clean", "accuracy: deformed", "accuracy: far"]
tlog = pickle.load(open(PKL, "rb"))["teachers"]["G"]
base = pickle.load(open(PKL.replace(".pkl", "b.pkl"), "rb"))["baseline"][0]
log_prior = np.log(np.bincount(d.y[lab], minlength=11) / lab.sum())
posthoc = lambda oof: {f"+{t:g}": round(balanced_accuracy_score(d.y[lab], (np.log(oof[lab] + 1e-12) - t * log_prior).argmax(1)), 3) for t in (0, 0.5, 1.0)}

t = time.time()
toof = distill.teacher_nested(d.G, d.y, fold, CFG, seed=0)
print(f"nested teachers ({time.time()-t:.0f}s)", flush=True)
out = {"teacher_oof": toof}
for scfg in [distill.StudentConfig(lam=2.0, T=2.0),
             distill.StudentConfig(lam=0.0, beta=1.0),
             distill.StudentConfig(lam=2.0, T=2.0, beta=1.0),
             distill.StudentConfig(lam=2.0, T=2.0, lam_lab=1.0),
             distill.StudentConfig(lam=2.0, T=2.0, lam_lab=1.0, beta=1.0)]:
    t = time.time()
    m, oof = distill.run_student_cv(d.E, d.y, groups, fold, tlog, CFG, scfg, seeds=(0,), G=d.G, teacher_oof=toof)
    diff = m["balanced accuracy"].to_numpy() - base["balanced accuracy"].to_numpy()
    print(f"{scfg.name()} ({time.time()-t:.0f}s):", m[cols].mean().round(3).to_dict(),
          "| paired %+.3f (%d/5)" % (diff.mean(), (diff > 0).sum()), "| pooled OOF, extra tau:", posthoc(oof), flush=True)
    out[scfg.name()] = (m, oof)
pickle.dump(out, open(PKL.replace(".pkl", "c.pkl"), "wb"))
