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
from dataclasses import dataclass

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


def fit_student(X_lab, y_lab, X_unl, Q_unl, w_unl, cfg, scfg, seed=0, Q_lab=None, G_lab=None, G_unl=None):
    """One student MLP on the main features. Loss terms, each weighted by its share of the batch:

    - labelled rows: cfg.loss (logit-adjusted cross-entropy) on the true label;
    - labelled rows, if scfg.lam_lab: lam_lab x soft cross-entropy against out-of-fold teacher targets Q_lab;
    - unlabelled rows: scfg.lam x soft cross-entropy against teacher targets Q_unl (rows weighted by w_unl);
    - all rows, if scfg.beta: beta x mean squared error of a second head predicting the standardized G.

    With scfg.adjust the soft losses use the same prior shift as logit-adjusted cross-entropy: softmax(logits + shift)
    is matched to the teacher's training-prior posterior, so every term trains the raw logits to be uniform-prior
    scores. Without the shift, the unlabelled rows (which follow the training class mix, ~38% class 0) act like plain
    cross-entropy and pull the student back towards class 0.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X_all = np.vstack([X_lab, X_unl])
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
    n_lab, n = len(X_lab), len(X_all)
    steps_per_epoch = int(np.ceil(n / cfg.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch)
    X_t = torch.as_tensor(X_all)
    yl = torch.as_tensor(y_lab, dtype=torch.long)
    Qu, wu = torch.as_tensor(Q_unl), torch.as_tensor(w_unl)
    Ql = torch.as_tensor(Q_lab) if Q_lab is not None else None
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
            il, iu = idx[idx < n_lab], idx[idx >= n_lab] - n_lab
            loss = 0.0
            if len(il):
                share = len(il) / len(idx)
                loss = loss + share * hard_loss(logits[is_lab], yl[il])
                if Ql is not None and scfg.lam_lab:
                    loss = loss + share * scfg.lam_lab * soft_ce(logits[is_lab], Ql[il]).mean()
            if len(iu) and scfg.lam:
                share = len(iu) / len(idx)
                loss = loss + share * scfg.lam * (wu[iu] * soft_ce(logits[~is_lab], Qu[iu])).mean()
            if scfg.beta:
                loss = loss + scfg.beta * F.mse_loss(g_pred, G_z[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model


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
