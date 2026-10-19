"""Phases 3-4: a teacher MLP on the supplementary features passes what it knows to a student MLP on the main features.

The supplementary features G exist only for training rows, so only the student can predict the test set.
This is learning using privileged information via generalized distillation (Lopez-Paz et al., 2016).
The student can learn from the teacher through three channels (see StudentConfig and fit_student):
- soft labels on the unlabelled rows (pseudo-labelling);
- soft labels on the labelled rows, from out-of-fold teacher predictions;
- an auxiliary head that predicts G itself from the main features.

Leakage control: arrays cover all 2076 training rows, with fold = -1 for unlabelled rows. For validation fold k,
the teachers are trained only on labelled rows outside fold k, and the student only sees those rows plus the
unlabelled rows, so nothing about fold k's labels reaches either model.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import mlp
from .cv import N_FOLDS, evaluate
from .data import N_CLASSES


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def teacher_cv(X, y, groups, fold, cfg, seeds=(0,)):
    """Teacher trained on the labelled rows outside each fold k.

    Returns the teacher's metrics on each fold k, and teacher logits for every training row per fold k
    (averaged over seeds). The logits of fold k are only ever used for the student that is validated on fold k.
    """
    labelled = fold >= 0
    metrics, logits = [], {}
    for k in range(N_FOLDS):
        tr, va = labelled & (fold != k), fold == k
        L = np.zeros((len(y), N_CLASSES))
        for s in seeds:
            model, _ = mlp.fit(X[tr], y[tr], cfg, seed=s)
            L += mlp.predict_logits(model, X) / len(seeds)
        logits[k] = L
        metrics.append({"fold": k, **evaluate(y[va], L[va].argmax(1), groups[va])})
    return pd.DataFrame(metrics), logits


def teacher_nested(X, y, fold, cfg, seed=0):
    """Out-of-fold teacher logits for the labelled training rows of each outer fold k (needed for lam_lab).

    For outer fold k and every other fold j, a teacher trained on the labelled rows outside folds k and j predicts
    fold j. Fold k's labels are never used, so the student validated on fold k stays leak-free.
    """
    labelled = fold >= 0
    out = {}
    for k in range(N_FOLDS):
        L = np.zeros((len(y), N_CLASSES))
        for j in range(N_FOLDS):
            if j == k:
                continue
            tr = labelled & (fold != k) & (fold != j)
            model, _ = mlp.fit(X[tr], y[tr], cfg, seed=seed)
            L[fold == j] = mlp.predict_logits(model, X[fold == j])
        out[k] = L
    return out


@dataclass(frozen=True)
class StudentConfig:
    lam: float = 1.0          # weight of the soft loss on teacher-labelled (unlabelled) rows
    T: float = 1.0            # temperature applied to the teacher logits (higher = softer targets)
    threshold: float = 0.0    # use teacher labels only where the teacher's top-class probability is at least this
    hard: bool = False        # use the teacher's top class as a hard label instead of its soft distribution
    adjust: bool = True       # apply the logit-adjustment prior shift in the soft losses too (see fit_student)
    lam_lab: float = 0.0      # weight of a soft loss on labelled rows, against out-of-fold teacher logits
    beta: float = 0.0         # weight of an auxiliary head that predicts the standardized supplementary features

    def name(self):
        kind = "hard" if self.hard else f"soft T={self.T:g}"
        extra = "" if self.adjust else ", no prior shift"
        if self.lam_lab:
            extra += f", lam_lab={self.lam_lab:g}"
        if self.beta:
            extra += f", G-head beta={self.beta:g}"
        return f"student ({kind}, lam={self.lam:g}, threshold={self.threshold:g}{extra})"


def prior_shift(cfg, y_lab):
    """tau * log(class prior of the labelled training rows) for logit-adjusted training, else zeros."""
    if cfg.loss != "logit_adjusted":
        return np.zeros(N_CLASSES, dtype=np.float32)
    counts = np.bincount(y_lab, minlength=N_CLASSES)
    return (cfg.tau * np.log(counts / counts.sum())).astype(np.float32)


def teacher_targets(teacher_logits, scfg, shift):
    """Soft targets for teacher-labelled rows, and a 0/1 weight per row (the confidence threshold).

    A logit-adjusted teacher's raw logits assume a uniform prior. Adding `shift` turns them back into its estimate of
    the training-prior posterior, which is what the student's shifted logits are matched against (see fit_student).
    """
    z = teacher_logits + (shift if scfg.adjust else 0.0)
    weight = (softmax(z).max(axis=1) >= scfg.threshold).astype(np.float32)
    q = softmax(z, scfg.T)
    if scfg.hard:
        q = np.eye(N_CLASSES)[q.argmax(axis=1)]
    return q.astype(np.float32), weight


class MultiTaskMLP(mlp.MLP):
    """The Phase 1 MLP plus a second head on the last hidden layer that predicts the supplementary features."""

    def __init__(self, mean, std, cfg, g_dim):
        super().__init__(mean, std, cfg)
        self.g_head = nn.Linear(cfg.hidden[-1], g_dim)

    def heads(self, x):
        h = self.net[:-1]((x - self.mean) / self.std)
        return self.net[-1](h), self.g_head(h)


def fit_student(X_lab, y_lab, X_unl, Q_unl, w_unl, cfg, scfg, seed=0, Q_lab=None, G_lab=None, G_unl=None,
                X_ext=None, Q_ext=None, w_ext=None, lam_ext=0.0):
    """One student MLP on the main features. Loss terms, each weighted by its share of the batch:

    - labelled rows: cfg.loss (logit-adjusted cross-entropy) on the true label;
    - labelled rows, if scfg.lam_lab: lam_lab x soft cross-entropy against out-of-fold teacher targets Q_lab;
    - unlabelled rows: scfg.lam x soft cross-entropy against teacher targets Q_unl (rows weighted by w_unl);
    - all rows that have G, if scfg.beta: beta x mean squared error of a second head predicting the standardized G;
    - extra rows without G or label (self-training on test rows), if lam_ext: lam_ext x soft cross-entropy against
      Q_ext (rows weighted by w_ext).

    With scfg.adjust the soft losses use the same prior shift as logit-adjusted cross-entropy: softmax(logits + shift)
    is matched to the teacher's training-prior posterior, so every term trains the raw logits to be uniform-prior
    scores. Without the shift, the unlabelled rows (which follow the training class mix, ~38% class 0) act like plain
    cross-entropy and pull the student back towards class 0.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_ext = 0 if X_ext is None else len(X_ext)
    X_all = np.vstack([X_lab, X_unl] + ([X_ext] if n_ext else []))
    std = X_all.std(axis=0)
    mean, std = X_all.mean(axis=0), np.where(std > 0, std, 1.0)
    if scfg.beta:
        G_all = np.vstack([G_lab, G_unl])
        G_z = torch.as_tensor((G_all - G_all.mean(axis=0)) / (G_all.std(axis=0) + 1e-8), dtype=torch.float32)
        model = MultiTaskMLP(mean, std, cfg, G_all.shape[1])
    else:
        model = mlp.MLP(mean, std, cfg)
    hard_loss = mlp.make_loss(cfg, y_lab)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_lab, n_g, n = len(X_lab), len(X_lab) + len(X_unl), len(X_all)     # rows: labelled | unlabelled | extra
    steps_per_epoch = int(np.ceil(n / cfg.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch)
    X_t = torch.as_tensor(X_all)
    yl = torch.as_tensor(y_lab, dtype=torch.long)
    Qu, wu = torch.as_tensor(Q_unl), torch.as_tensor(w_unl)
    Ql = torch.as_tensor(Q_lab) if Q_lab is not None else None
    Qe, we = (torch.as_tensor(Q_ext), torch.as_tensor(w_ext)) if n_ext else (None, None)
    shift = torch.as_tensor(prior_shift(cfg, y_lab)) if scfg.adjust else torch.zeros(N_CLASSES)

    def soft_ce(logits, q):
        return -(q * F.log_softmax(logits + shift, dim=1)).sum(dim=1)

    for _ in range(cfg.epochs):
        model.train()
        for idx in np.array_split(rng.permutation(n), steps_per_epoch):
            if scfg.beta:
                logits, g_pred = model.heads(X_t[idx])
            else:
                logits = model(X_t[idx])
            is_lab = torch.as_tensor(idx < n_lab)
            is_unl = torch.as_tensor((idx >= n_lab) & (idx < n_g))
            is_ext = torch.as_tensor(idx >= n_g)
            il, iu, ie = idx[idx < n_lab], idx[(idx >= n_lab) & (idx < n_g)] - n_lab, idx[idx >= n_g] - n_g
            loss = 0.0
            if len(il):
                share = len(il) / len(idx)
                loss = loss + share * hard_loss(logits[is_lab], yl[il])
                if Ql is not None and scfg.lam_lab:
                    loss = loss + share * scfg.lam_lab * soft_ce(logits[is_lab], Ql[il]).mean()
            if len(iu) and scfg.lam:
                share = len(iu) / len(idx)
                loss = loss + share * scfg.lam * (wu[iu] * soft_ce(logits[is_unl], Qu[iu])).mean()
            if len(ie) and lam_ext:
                share = len(ie) / len(idx)
                loss = loss + share * lam_ext * (we[ie] * soft_ce(logits[is_ext], Qe[ie])).mean()
            if scfg.beta:
                has_g = idx < n_g
                if has_g.any():
                    share = has_g.sum() / len(idx)
                    loss = loss + share * scfg.beta * F.mse_loss(g_pred[torch.as_tensor(has_g)], G_z[idx[has_g]])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model


def fit_final(E, G, y, fold, cfg, scfg, teacher_seeds=(0, 1, 2), student_seeds=(0, 1, 2, 3, 4),
              X_test=None, lam_ext=0.0, T_ext=1.0, threshold_ext=0.0):
    """Final models trained on all training rows.

    - Teacher logits for the unlabelled rows: teachers on G trained on all labelled rows (averaged over seeds).
    - Teacher logits for the labelled rows (if scfg.lam_lab): out-of-fold over the fixed folds, so they are not
      just the memorized labels.
    - Students: one per seed, on E of all labelled + unlabelled rows.
    - If lam_ext > 0 (self-training): the students' averaged predictions on X_test become soft targets, and a second
      round of students is trained with the test rows as extra rows (see run_selftrain_cv).
    """
    labelled, unlabelled = fold >= 0, y < 0
    L_unl = np.zeros((unlabelled.sum(), N_CLASSES))
    for s in teacher_seeds:
        teacher, _ = mlp.fit(G[labelled], y[labelled], cfg, seed=s)
        L_unl += mlp.predict_logits(teacher, G[unlabelled]) / len(teacher_seeds)
    L_lab = np.zeros((len(y), N_CLASSES))
    if scfg.lam_lab:
        for k in range(N_FOLDS):
            for s in teacher_seeds:
                teacher, _ = mlp.fit(G[labelled & (fold != k)], y[labelled & (fold != k)], cfg, seed=s)
                L_lab[fold == k] += mlp.predict_logits(teacher, G[fold == k]) / len(teacher_seeds)
    shift = prior_shift(cfg, y[labelled])
    Q_unl, w_unl = teacher_targets(L_unl, scfg, shift)
    Q_lab = teacher_targets(L_lab[labelled], scfg, shift)[0] if scfg.lam_lab else None
    common = dict(Q_lab=Q_lab, G_lab=G[labelled], G_unl=G[unlabelled])
    students = [fit_student(E[labelled], y[labelled], E[unlabelled], Q_unl, w_unl, cfg, scfg, s, **common)
                for s in student_seeds]
    info = {"unlabelled": L_unl, "labelled_oof": L_lab[labelled]}
    if lam_ext:
        L_test = np.mean([mlp.predict_logits(m, X_test) for m in students], axis=0)
        Q_ext, w_ext = teacher_targets(L_test, replace(scfg, T=T_ext, threshold=threshold_ext, hard=False), shift)
        students = [fit_student(E[labelled], y[labelled], E[unlabelled], Q_unl, w_unl, cfg, scfg, s, **common,
                                X_ext=X_test, Q_ext=Q_ext, w_ext=w_ext, lam_ext=lam_ext) for s in student_seeds]
        info["stage1_test_logits"] = L_test
    return students, info


def run_student_cv(X, y, groups, fold, teacher_logits, cfg, scfg, seeds=(0,), G=None, teacher_oof=None):
    """Student CV on the fixed folds.

    teacher_logits: output of teacher_cv (one array per fold; used for the unlabelled rows).
    G: supplementary features (needed if scfg.beta). teacher_oof: output of teacher_nested (needed if scfg.lam_lab).
    """
    labelled, unlabelled = fold >= 0, y < 0
    metrics = []
    oof = np.zeros((len(y), N_CLASSES))
    for seed in seeds:
        for k in range(N_FOLDS):
            tr, va = labelled & (fold != k), fold == k
            shift = prior_shift(cfg, y[tr])
            Q_unl, w_unl = teacher_targets(teacher_logits[k][unlabelled], scfg, shift)
            Q_lab = teacher_targets(teacher_oof[k][tr], scfg, shift)[0] if scfg.lam_lab else None
            model = fit_student(X[tr], y[tr], X[unlabelled], Q_unl, w_unl, cfg, scfg, seed, Q_lab=Q_lab,
                                G_lab=G[tr] if scfg.beta else None, G_unl=G[unlabelled] if scfg.beta else None)
            p = mlp.predict_proba(model, X[va])
            oof[va] += p / len(seeds)
            metrics.append({"config": scfg.name(), "seed": seed, "fold": k, "teacher-labelled rows": int(w_unl.sum()),
                            **evaluate(y[va], p.argmax(1), groups[va])})
    return pd.DataFrame(metrics), oof


def run_selftrain_cv(X, y, groups, fold, teacher_logits, cfg, scfg, X_test, lam_ext=1.0, T_ext=1.0, threshold_ext=0.0,
                     seeds=(0,), G=None, teacher_oof=None):
    """Two-stage self-training (transductive), scored on the fixed folds.

    Stage 1 is the normal student. Stage 2 retrains it with extra rows that have neither label nor G: the
    validation fold and the test rows, with stage 1's predictions as soft targets. This mirrors the final model, where
    the test rows are the extra rows. The validation labels are only used for scoring.
    """
    labelled, unlabelled = fold >= 0, y < 0
    ext_cfg = replace(scfg, T=T_ext, threshold=threshold_ext, hard=False)
    metrics = []
    for seed in seeds:
        for k in range(N_FOLDS):
            tr, va = labelled & (fold != k), fold == k
            shift = prior_shift(cfg, y[tr])
            Q_unl, w_unl = teacher_targets(teacher_logits[k][unlabelled], scfg, shift)
            Q_lab = teacher_targets(teacher_oof[k][tr], scfg, shift)[0] if scfg.lam_lab else None
            common = dict(Q_lab=Q_lab, G_lab=G[tr] if scfg.beta else None, G_unl=G[unlabelled] if scfg.beta else None)
            stage1 = fit_student(X[tr], y[tr], X[unlabelled], Q_unl, w_unl, cfg, scfg, seed, **common)
            X_ext = np.vstack([X[va], X_test])
            Q_ext, w_ext = teacher_targets(mlp.predict_logits(stage1, X_ext), ext_cfg, shift)
            stage2 = fit_student(X[tr], y[tr], X[unlabelled], Q_unl, w_unl, cfg, scfg, seed, **common,
                                 X_ext=X_ext, Q_ext=Q_ext, w_ext=w_ext, lam_ext=lam_ext)
            for stage, model in [("stage 1", stage1), ("stage 2", stage2)]:
                p = mlp.predict_proba(model, X[va])
                metrics.append({"stage": stage, "seed": seed, "fold": k, "extra rows used": int(w_ext.sum()),
                                **evaluate(y[va], p.argmax(1), groups[va])})
    return pd.DataFrame(metrics)
