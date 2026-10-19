# Experiment log

One entry per experiment, as the challenge rules require. Every entry uses the fixed folds in `folds/cv_folds_v1.json`
(Phase 0), unless the entry says otherwise.

- **Main metric:** cross-validated **balanced accuracy**, our stand-in for the hidden-weight accuracy.
- Also reported: plain accuracy and accuracy per group (clean / deformed / far).
- "Kaggle" is the public leaderboard score, filled in after uploading.

| # | Date | Experiment | Features | Model / loss | CV bal. acc. | CV acc. | clean / deformed / far | Kaggle | Where |
|---|---|---|---|---|---|---|---|---|---|
| 00 | 2026-10-18 | Linear probe (diagnostic only, not a submission) | E | ridge, α = 100 | 0.632 | 0.775 | 0.928 / 0.776 / 0.624 | — | `phase1_baseline_mlp.ipynb` §2 |
| 02a | 2026-10-18 | MLP, plain cross-entropy | E | MLP B, CE | 0.657 | 0.775 | 0.928 / 0.778 / 0.608 | — | §4 |
| 02b | 2026-10-18 | MLP, class-weighted CE | E | MLP B, weighted CE | 0.679 | 0.775 | 0.935 / 0.778 / 0.601 | — | §4 |
| **02c** | 2026-10-18 | **Phase 1 baseline** | E | MLP B, logit-adjusted CE (τ = 1) | **0.682** | **0.775** | 0.927 / 0.781 / 0.592 | _to fill_ | §4–5 |

MLP B = 768 → 512 → 256 → 11, GELU, dropout 0.5, input dropout 0.2, AdamW (lr 1e-3, weight decay 0.05), batch 64,
40 epochs with cosine decay, standardization fitted on the training part. Rows 02a–c are means over 3 seeds × 5 folds.

---

## 01 — Phase 0: validation protocol (2026-10-18)

- **Groups** for every training row: `clean` (is_deformed = 0); `far` (deformed and PC1 > 15 in a 2-component PCA of the
  standardized training features, with PC1 oriented from the clean mean towards the deformed mean); `deformed` (all other
  deformed rows). Counts: 208 / 225 / 1643. 55 test rows fall in the far group.
- **Folds:** 5-fold `StratifiedKFold` (shuffle, seed 0) on the 1284 labelled rows, stratified on class × group.
  Unlabelled rows get fold −1 and are never scored. Saved to `folds/cv_folds_v1.json` by `scripts/make_folds.py`.
  Code: `ie643/cv.py`.
- **Metrics:** balanced accuracy (main), accuracy, accuracy per group (`ie643.cv.evaluate`).
- **Noise level:** the fold-to-fold std of balanced accuracy is about 0.03. Differences below about 0.02 between single
  runs are noise, so comparisons use 3 seeds and run-by-run (paired) differences.

## 02 — Phase 1: baseline MLP on E (2026-10-18)

Notebook: `phase1_baseline_mlp.ipynb`. Code: `ie643/mlp.py`.

**Pilot** (scratch run, not in the notebook): setting A (below) with 150 epochs and 1 seed reached balanced accuracy 0.655,
against 0.662 at 40 epochs. Training loss was ~0 by epoch 10, so 40 epochs is used from here on.

**Regularization sweep** (CE, 1 seed, 40 epochs): six settings, all within noise of each other.

| Setting | Balanced acc. | Acc. |
|---|---|---|
| A: 512-256, dropout .3, input .1, wd .01 | 0.662 | 0.774 |
| **B: 512-256, dropout .5, input .2, wd .05 (kept)** | **0.664** | **0.782** |
| C: 256, dropout .5, input .2, wd .05 | 0.660 | 0.778 |
| D: 512-256, dropout .5, input .4, wd .1 | 0.652 | 0.772 |
| E: 1024-512, dropout .5, input .2, wd .05 | 0.659 | 0.772 |
| F: as B, lr 3e-4 | 0.652 | 0.774 |

**Loss comparison** (setting B, 3 seeds × 5 folds): see the table at the top.
- Logit adjustment beats CE by +0.026 balanced accuracy, in 14 of 15 paired runs, with no change in plain accuracy.
- The gains come from the rare classes: class 10 recall 0.50 → 0.60, class 2 0.44 → 0.52, class 1 0.65 → 0.72.

**Final model:** 5 seeds trained on all 1284 labelled rows, test probabilities averaged.
- Weights: `checkpoints/phase1_baseline/` (git-ignored, kept for the Drive hand-in).
- Submission: `submissions/phase1_mlp_logit_adjusted.csv`.

**Findings**
1. Capacity and regularization are not the bottleneck; the information is. Next: supplementary-feature teacher (Phase 3) and unlabelled rows (Phase 4).
2. The networks overfit within a few epochs and become overconfident. Calibrate them before distillation or pseudo-labelling.
3. **Label shift:** on test, the model predicts class 0 for only 12.2% of rows, while on validation folds it reproduces the training share (35.1% vs 35.0%). The linear probe shows the same drop (40.3% → 21.6%). The test set very likely has far fewer class-0 rows, possibly close to balanced. Balanced accuracy is the right metric to track, and correcting for the class prior matters.
4. The far group is still the weakest part (0.59 accuracy).
