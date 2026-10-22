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
| 02c | 2026-10-18 | Phase 1 baseline | E | MLP B, logit-adjusted CE (τ = 1) | 0.682 | 0.775 | 0.927 / 0.781 / 0.592 | 0.6165 | §4–5 |
| 03t | 2026-10-19 | Teacher (cannot predict test; 1 seed) | G | MLP B, logit-adjusted CE | 0.832 | 0.877 | 0.963 / 0.877 / 0.795 | — | `phase3_distillation.ipynb` §2 |
| **03** | 2026-10-19 | **Phase 3/4 student, all three channels** | E (+ G as teacher) | MLP B + G-head; pseudo-labels, labelled-row distillation | **0.708** | **0.792** | 0.949 / 0.800 / 0.595 | **0.6619** | §3–5 |

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
5. **Kaggle public score 0.6165 (2026-10-19).**
   - It is far below CV plain accuracy (0.775) and closer to CV balanced accuracy (0.682). So the leaderboard behaves like a class-balanced metric: either the test set is close to balanced, or the hidden weights favour the rare classes. We cannot tell which, but balanced accuracy is the right CV target either way, which supports finding 3.
   - The remaining gap of about 0.07 is partly noise. The public leaderboard scores only part of the 490 test rows, so its standard error is about ±0.03. The rest may come from hidden weights favouring hard classes (2, 8 and 10 have recall of only 0.5–0.6) or from the mild train/test shift (adversarial AUC 0.576).

## 03 — Phases 3 + 4: generalized distillation from G (2026-10-19)

Notebook: `phase3_distillation.ipynb`. Code: `ie643/distill.py`.

**Method:** learning using privileged information via generalized distillation (Vapnik & Vashist 2009; Lopez-Paz et al. 2016).
- A teacher MLP on G, the 384 supplementary features, which exist only for training rows, passes what it knows to a student MLP on E. Only the student predicts test.
- Both use MLP setting B and logit-adjusted CE.
- Three channels from teacher to student:
  1. soft labels on the 792 unlabelled rows (T = 2, λ = 2);
  2. soft labels on the labelled rows from nested out-of-fold teachers (λ_lab = 1);
  3. a second head that predicts the standardized G from E on all 2076 rows (MSE, β = 1).
- Leakage control: for validation fold k, all teachers are trained without fold k's labels.

**Pilot path** (scratch runs, reproduced in the notebook):
- **Teacher:** G 0.832 vs E+G 0.808 balanced accuracy, so G alone is the teacher.
- **First students:** soft targets with no prior shift scored 0.674–0.687, against 0.691 for the seed-0 baseline.
  - Diagnosis: the unlabelled rows follow the training mix (39% class 0), and the soft loss had no logit adjustment, so it undid the adjustment.
  - Fix: use the same prior shift in every soft loss, with targets from the teacher's training-prior posterior.
- **After the fix:**
  - pseudo-labels 0.699;
  - G-head 0.702;
  - pseudo-labels + labelled-row distillation 0.704;
  - all three channels 0.707.
- **Extra post-hoc prior correction** (τ + 0.5 or + 1 at prediction time): it did not help consistently on CV, so it is not used.

**Confirmation** (3 seeds × 5 folds): student 0.708 vs baseline 0.682.
- Run by run: +0.026 (standard error 0.010), higher in 12 of 15 runs.
- Plain accuracy: 0.775 → 0.792.
- Per-class recall: class 2 0.52 → 0.62, class 10 0.60 → 0.72, class 1 0.72 → 0.63.
- The far group is unchanged (0.592 → 0.595).

**Final model:** teachers on all labelled G (3 seeds), out-of-fold teacher logits for the labelled rows, and 5 student seeds on all 2076 rows.
- Weights: `checkpoints/phase3_student/`.
- Submission: `submissions/phase3_student_all_channels.csv`.
- It agrees with the Phase 1 submission on 84.5% of test rows.

**Findings**
1. Distillation gives a real but modest gain. The student cannot recover most of what G knows (teacher 0.832) from E alone.
2. Prior consistency across loss terms is essential under label imbalance.
3. The far group (about 11% of test) is still at about 0.59 accuracy, while the teacher scores 0.795 there. It is the largest remaining gap.
4. **Kaggle public score 0.6619 (2026-10-19), up from 0.6165.** The leaderboard gain (+0.045) is larger than the CV gain (+0.026), so CV improvements carry over to the leaderboard. Rank 42; the top score is 0.7226.

## 04 — Phase 5: student tuning, a test-row experiment, the G ceiling, and the final ensembles (2026-10-19)

Scripts and console logs: `experiments/` (see its README). Final training: `scripts/train_final.py`.

**Student settings sweep** (e5, 1 seed; compared run by run with the all-channel student at 0.707):
- epochs 60/25 and width 1024-512 are within ±0.003 (noise);
- less dropout: −0.012;
- λ_lab = 3: −0.007;
- T = 4: +0.004, but accuracy 0.769 and far group 0.542;
- β = 3: +0.007;
- τ = 1.5: +0.008.

**Confirmation over 3 seeds × 5 folds** (e7, run by run vs the current student at 0.708):

| Setting | Balanced acc. | Accuracy | Far group |
|---|---|---|---|
| β = 3 | 0.714 (+0.005, se 0.004) | 0.793 (+0.001) | 0.607 (+0.012) |
| τ = 1.5 | 0.712 (+0.004) | 0.777 (−0.016) | 0.550 (−0.046) |
| β = 3 + τ = 1.5 | 0.715 (+0.007) | 0.778 (−0.015) | 0.550 (−0.045) |

- β = 3 is adopted: a small gain at no cost.
- τ = 1.5 trades class-0 and far-group accuracy for rare-class recall. CV cannot decide it, because that depends on the test class mix, so both versions are submitted.

**Self-training on the test rows** (e6). **Experiment only, not used in any submission.**
- Method: stage 2 retrains the student with the validation fold plus the 490 test rows as unlabelled rows, labelled by stage 1. Validation labels are used only for scoring.
- Result: worse than stage 1 in every variant: −0.012 (all rows), −0.007 (confidence ≥ 0.7), −0.010 (λ = 2, confidence ≥ 0.5).
- Decision: kept out of all submissions on principle (training on test inputs), and it would not have helped anyway.

**How much of G can E reach?** (e8)
- A linear map from E predicts G with mean R² 0.47 overall, but only **0.05 inside the far group**. The deformation has destroyed the G-related information in those rows' E vectors, which explains why no distillation channel helped the far group.
- Classifying through predicted G scores 0.657–0.667 balanced accuracy, below the student (0.708) and even the baseline (0.682).
- Conclusion: the part of G that E can reach is mostly information E already gives. More G-transfer work is unlikely to pay off.

**Final ensembles** (fully inductive: Train_Data + supplementary only; 3 teacher seeds, 10 student seeds):
- A: β = 3, τ = 1 → `submissions/final_A_beta3_tau1.csv`
- B: β = 3, τ = 1.5 → `submissions/final_B_beta3_tau1.5.csv`

**Kaggle scores (2026-10-19):** A (β = 3) 0.6512; B (β = 3, τ = 1.5) 0.6616. The previous best is 0.6619.
- A and B change only 14 and 29 test predictions, so these differences are within public-leaderboard noise.
- Neither is a meaningful change. Final selections: the 0.6619 submission and B.

## 05 — Teammate's labels for the 72 clean unlabelled rows (2026-10-19)

- **Source:** files from the teammate's work in `ie-643-2026-programming-challenge/teammate_labels/` (logistic regression and 5-NN cosine on 118 PCs of clean rows).
- **Agreement:** the two methods agree on 93.1% of rows. Our G-teacher agrees with them on 93–94%; where it disagrees, it is almost 100% confident.
- **Experiment** (e9; 3 seeds × 5 folds, compared run by run with the β = 3 student at 0.714): move the 72 rows from "unlabelled, teacher soft labels" to "labelled, hard teammate label".
  - Provided labels as-is: 0.711 (−0.002, se 0.003).
  - Same recipe regenerated inside each fold, which is leak-free: 0.710 (−0.004).
  - No gain: the student already learns these rows from the teacher. **Not used.**
- **Rules note:** logistic regression is a single-layer softmax network, but k-NN is not an MLP/CNN. If such labels were ever used in a submission, only an MLP-based version would be acceptable.
