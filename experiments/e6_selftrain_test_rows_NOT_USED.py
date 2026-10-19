import sys, time, pickle
from dataclasses import replace
sys.path.insert(0, sys.argv[1]); PKL = sys.argv[2]
import numpy as np, pandas as pd
from ie643 import cv, data, mlp, distill
from ie643.cv import evaluate

d = data.load()
groups, fold = cv.load_folds(d.train_id)
lab, unl = fold >= 0, d.y < 0
CFG = mlp.Config(epochs=40, dropout=0.5, input_dropout=0.2, weight_decay=0.05, loss="logit_adjusted")
S = distill.StudentConfig(lam=2.0, T=2.0, lam_lab=1.0, beta=1.0)
cols = ["balanced accuracy", "accuracy", "accuracy: clean", "accuracy: deformed", "accuracy: far"]
tlog = pickle.load(open(PKL, "rb"))["teachers"]["G"]
toof = pickle.load(open(PKL.replace(".pkl", "c.pkl"), "rb"))["teacher_oof"]
EXT = {"lam_ext 1, all rows": (1.0, 1.0, 0.0), "lam_ext 1, conf >= 0.7": (1.0, 1.0, 0.7), "lam_ext 2, conf >= 0.5": (2.0, 1.0, 0.5)}
rows = []
t0 = time.time()
for k in range(cv.N_FOLDS):
    tr, va = lab & (fold != k), fold == k
    shift = distill.prior_shift(CFG, d.y[tr])
    Q_unl, w_unl = distill.teacher_targets(tlog[k][unl], S, shift)
    Q_lab = distill.teacher_targets(toof[k][tr], S, shift)[0]
    common = dict(Q_lab=Q_lab, G_lab=d.G[tr], G_unl=d.G[unl])
    stage1 = distill.fit_student(d.E[tr], d.y[tr], d.E[unl], Q_unl, w_unl, CFG, S, 0, **common)
    rows.append({"variant": "stage 1 (current student)", "fold": k, **evaluate(d.y[va], mlp.predict_proba(stage1, d.E[va]).argmax(1), groups[va])})
    X_ext = np.vstack([d.E[va], d.F])
    L_ext = mlp.predict_logits(stage1, X_ext)
    for name, (lam_ext, T_ext, thr) in EXT.items():
        Q_ext, w_ext = distill.teacher_targets(L_ext, replace(S, T=T_ext, threshold=thr, hard=False), shift)
        stage2 = distill.fit_student(d.E[tr], d.y[tr], d.E[unl], Q_unl, w_unl, CFG, S, 0, **common,
                                     X_ext=X_ext, Q_ext=Q_ext, w_ext=w_ext, lam_ext=lam_ext)
        rows.append({"variant": name, "fold": k, "extra rows used": int(w_ext.sum()),
                     **evaluate(d.y[va], mlp.predict_proba(stage2, d.E[va]).argmax(1), groups[va])})
    print(f"fold {k} done ({time.time()-t0:.0f}s)", flush=True)
res = pd.DataFrame(rows)
print(res.groupby("variant", sort=False)[cols + ["extra rows used"]].mean().round(3).to_string())
piv = res.pivot_table(index="fold", columns="variant", values="balanced accuracy")
for name in EXT:
    diff = piv[name] - piv["stage 1 (current student)"]
    print(f"{name}: paired vs stage 1 {diff.mean():+.3f} (se {diff.std()/np.sqrt(len(diff)):.3f}, {(diff > 0).sum()}/5 higher)")
pickle.dump(res, open(PKL.replace(".pkl", "_selftrain.pkl"), "wb"))
