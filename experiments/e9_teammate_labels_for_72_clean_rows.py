"""Student (beta=3) with the 72 clean unlabelled rows moved into the labelled set, using teammate-style labels."""
import sys, time, pickle
from dataclasses import replace
sys.path.insert(0, sys.argv[1]); PKL = sys.argv[2]
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ie643 import cv, data, mlp, distill
from ie643.cv import evaluate

d = data.load()
groups, fold = cv.load_folds(d.train_id)
lab, unl = fold >= 0, d.y < 0
CFG = mlp.Config(epochs=40, dropout=0.5, input_dropout=0.2, weight_decay=0.05, loss="logit_adjusted")
S = distill.StudentConfig(lam=2.0, T=2.0, lam_lab=1.0, beta=3.0)
cols = ["balanced accuracy", "accuracy", "accuracy: clean", "accuracy: deformed", "accuracy: far"]
tlog = pickle.load(open(PKL, "rb"))["teachers"]["G"]
toof = pickle.load(open(PKL.replace(".pkl", "c.pkl"), "rb"))["teacher_oof"]
ref = pickle.load(open(PKL.replace(".pkl", "_confirm_tuned.pkl"), "rb"))["beta 3"][0]

tm = pd.read_csv("ie-643-2026-programming-challenge/teammate_labels/pseudo_labels_clean_unlabelled_logreg_pca118.csv")
pos = pd.Series(np.arange(len(d.train_id)), index=d.train_id).loc[tm["sample_id"]].to_numpy()   # rows of the 72
provided = tm["predicted_label"].to_numpy()
clean_lab = lab & ~d.deformed

def regenerate(k):
    # teammate recipe (standardize -> PCA 99% -> logistic regression), fitted on clean labelled rows outside fold k
    fit = clean_lab & (fold != k)
    sc = StandardScaler().fit(d.E[fit]); pca = PCA(n_components=0.99, svd_solver="full").fit(sc.transform(d.E[fit]))
    clf = LogisticRegression(max_iter=5000).fit(pca.transform(sc.transform(d.E[fit])), d.y[fit])
    return clf.predict(pca.transform(sc.transform(d.E[pos])))

rest_unl = unl.copy(); rest_unl[pos] = False
rows = []
t0 = time.time()
for variant in ["provided labels (as-is)", "labels regenerated per fold"]:
    for seed in (0, 1, 2):
        for k in range(cv.N_FOLDS):
            tr, va = lab & (fold != k), fold == k
            y_ext = provided if variant.startswith("provided") else regenerate(k)
            X_lab = np.vstack([d.E[tr], d.E[pos]]); y_lab = np.r_[d.y[tr], y_ext]
            shift = distill.prior_shift(CFG, y_lab)
            Q_unl, w_unl = distill.teacher_targets(tlog[k][rest_unl], S, shift)
            Q_lab = distill.teacher_targets(np.vstack([toof[k][tr], tlog[k][pos]]), S, shift)[0]
            model = distill.fit_student(X_lab, y_lab, d.E[rest_unl], Q_unl, w_unl, CFG, S, seed, Q_lab=Q_lab,
                                        G_lab=np.vstack([d.G[tr], d.G[pos]]), G_unl=d.G[rest_unl])
            p = mlp.predict_proba(model, d.E[va])
            rows.append({"config": variant, "seed": seed, "fold": k, **evaluate(d.y[va], p.argmax(1), groups[va])})
    print(f"{variant} done ({time.time()-t0:.0f}s)", flush=True)
res = pd.concat([ref.assign(config="current student (beta 3)"), pd.DataFrame(rows)])
print(res.groupby("config", sort=False)[cols].mean().round(3).to_string())
for variant in ["provided labels (as-is)", "labels regenerated per fold"]:
    m = res[res.config == variant].sort_values(["seed", "fold"])
    r = ref.sort_values(["seed", "fold"])
    for metric in ["balanced accuracy", "accuracy"]:
        diff = m[metric].to_numpy() - r[metric].to_numpy()
        print(f"{variant} vs current, {metric}: {diff.mean():+.3f} (se {diff.std(ddof=1)/np.sqrt(len(diff)):.3f}, {(diff > 0).sum()}/15 higher)")
