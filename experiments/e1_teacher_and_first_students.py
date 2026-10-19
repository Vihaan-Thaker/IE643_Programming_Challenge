import sys, time, pickle
from dataclasses import replace
sys.path.insert(0, sys.argv[1]); OUT = sys.argv[2]
import numpy as np, pandas as pd
from ie643 import cv, data, mlp, distill

d = data.load()
groups, fold = cv.load_folds(d.train_id)
lab, unl = d.labelled, ~d.labelled
CFG = mlp.Config(epochs=40, dropout=0.5, input_dropout=0.2, weight_decay=0.05, loss="logit_adjusted")
cols = ["balanced accuracy", "accuracy", "accuracy: clean", "accuracy: deformed", "accuracy: far"]
res = {}

t = time.time()
base, _, _ = mlp.run_cv(d.E[lab], d.y[lab], groups[lab], fold[lab], CFG, seeds=(0,))
res["baseline"] = base
print(f"baseline seed0 ({time.time()-t:.0f}s):", base[cols].mean().round(3).to_dict(), flush=True)

teachers = {}
for name, X in [("G", d.G), ("E+G", np.hstack([d.E, d.G]))]:
    t = time.time()
    m, logits = distill.teacher_cv(X, d.y, groups, fold, CFG, seeds=(0,))
    teachers[name] = logits; res[f"teacher {name}"] = m
    print(f"teacher {name} ({time.time()-t:.0f}s):", m[cols].mean().round(3).to_dict(), flush=True)
    q = distill.softmax(logits[0][unl])
    print("   pseudo-label class mix on unlabelled rows (fold-0 teacher):", np.round(np.bincount(q.argmax(1), minlength=11) / unl.sum(), 3),
          "| mean top prob %.2f, share >= 0.7: %.2f" % (q.max(1).mean(), (q.max(1) >= 0.7).mean()), flush=True)

variants = [
    ("G", distill.StudentConfig(lam=1.0, T=1.0)),
    ("G", distill.StudentConfig(lam=1.0, T=2.0)),
    ("G", distill.StudentConfig(lam=0.5, T=1.0)),
    ("G", distill.StudentConfig(lam=1.0, hard=True, threshold=0.7)),
    ("E+G", distill.StudentConfig(lam=1.0, T=1.0)),
]
for tname, scfg in variants:
    t = time.time()
    m, _ = distill.run_student_cv(d.E, d.y, groups, fold, teachers[tname], CFG, scfg, seeds=(0,))
    key = f"teacher {tname} -> {scfg.name()}"
    res[key] = m
    diff = m["balanced accuracy"].to_numpy() - base["balanced accuracy"].to_numpy()
    print(f"{key} ({time.time()-t:.0f}s, {m['teacher-labelled rows'].iloc[0]} rows):", m[cols].mean().round(3).to_dict(),
          "| paired bal-acc diff vs baseline %+.3f (%d/5 folds higher)" % (diff.mean(), (diff > 0).sum()), flush=True)

pickle.dump({"res": res, "teachers": teachers}, open(OUT, "wb"))
print("saved", OUT)
