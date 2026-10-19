"""Phase 1: a plain MLP classifier on the 768 main features, and a cross-validation runner."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from .cv import N_FOLDS, evaluate
from .data import N_CLASSES

LOSSES = ("ce", "weighted_ce", "logit_adjusted")


@dataclass(frozen=True)
class Config:
    hidden: tuple = (512, 256)
    dropout: float = 0.3
    input_dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 100
    batch_size: int = 64
    loss: str = "ce"            # one of LOSSES
    tau: float = 1.0            # strength of the logit adjustment

    def name(self):
        return self.loss if self.loss != "logit_adjusted" else f"logit_adjusted (tau={self.tau:g})"


class MLP(nn.Module):
    """Standardization (statistics stored with the model) -> MLP -> 11 logits."""

    def __init__(self, mean, std, cfg, n_classes=N_CLASSES):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        layers, d = [nn.Dropout(cfg.input_dropout)], len(mean)
        for h in cfg.hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(cfg.dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def class_counts(y):
    return np.bincount(y, minlength=N_CLASSES).astype(np.float64)


def make_loss(cfg, y_train):
    counts = torch.as_tensor(class_counts(y_train), dtype=torch.float32)
    if cfg.loss == "ce":
        return lambda logits, t: F.cross_entropy(logits, t)
    if cfg.loss == "weighted_ce":                    # weight_c = n / (K * n_c): every class contributes equally
        w = counts.sum() / (N_CLASSES * counts)
        return lambda logits, t: F.cross_entropy(logits, t, weight=w)
    if cfg.loss == "logit_adjusted":                 # Menon et al. 2021: train on logits + tau * log(prior)
        shift = cfg.tau * torch.log(counts / counts.sum())
        return lambda logits, t: F.cross_entropy(logits + shift, t)
    raise ValueError(cfg.loss)


def fit(X, y, cfg, seed=0, X_val=None, y_val=None):
    """Train one MLP. Returns the model and a per-epoch history (validation metrics if X_val is given)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    std = X.std(axis=0)
    model = MLP(X.mean(axis=0), np.where(std > 0, std, 1.0), cfg)
    loss_fn = make_loss(cfg, y)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = int(np.ceil(len(X) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.long)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for idx in np.array_split(rng.permutation(len(X)), steps_per_epoch):
            loss = loss_fn(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        row = {"epoch": epoch, "train loss": total / len(X)}
        if X_val is not None:
            p = predict_proba(model, X_val)
            row["val loss"] = float(-np.log(p[np.arange(len(y_val)), y_val] + 1e-12).mean())
            row["val balanced accuracy"] = balanced_accuracy_score(y_val, p.argmax(1))
            row["val accuracy"] = float((p.argmax(1) == y_val).mean())
        history.append(row)
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_logits(model, X):
    model.eval()
    return model(torch.as_tensor(X, dtype=torch.float32)).numpy()


def predict_proba(model, X):
    return torch.softmax(torch.as_tensor(predict_logits(model, X)), dim=1).numpy()


def run_cv(X, y, groups, fold, cfg, seeds=(0, 1, 2)):
    """Train on 4 folds, evaluate on the 5th, for every fold and seed.

    Returns per-run metrics, per-epoch histories, and out-of-fold probabilities averaged over seeds.
    """
    metrics, histories = [], []
    oof = np.zeros((len(y), N_CLASSES))
    for seed in seeds:
        for k in range(N_FOLDS):
            tr, va = (fold >= 0) & (fold != k), fold == k
            model, hist = fit(X[tr], y[tr], cfg, seed=seed, X_val=X[va], y_val=y[va])
            p = predict_proba(model, X[va])
            oof[va] += p / len(seeds)
            metrics.append({"config": cfg.name(), "seed": seed, "fold": k, **evaluate(y[va], p.argmax(1), groups[va])})
            histories.append(hist.assign(config=cfg.name(), seed=seed, fold=k))
    return pd.DataFrame(metrics), pd.concat(histories, ignore_index=True), oof


def fit_ensemble(X, y, cfg, seeds=(0, 1, 2, 3, 4)):
    """Final models: train on all labelled rows with several seeds."""
    return [fit(X, y, cfg, seed=s)[0] for s in seeds]


def ensemble_proba(models, X):
    return np.mean([predict_proba(m, X) for m in models], axis=0)
