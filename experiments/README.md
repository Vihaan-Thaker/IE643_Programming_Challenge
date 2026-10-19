# Experiment scripts (Phases 3–5)

These are the scripts behind the decisions in `docs/experiment_log.md` entries 03–04, together with their console
output in `logs/`. All use the fixed folds (`folds/cv_folds_v1.json`) and the shared code in `ie643/`.
The final submission is produced separately by `scripts/train_final.py`.

## How to run

Run from the repository root with the project `.venv`. Each script takes the repository root (`.`) and the path of an
intermediate pickle in a cache folder (git-ignored), e.g.:

```bash
python experiments/e1_teacher_and_first_students.py . experiments/cache/pilot34.pkl
```

The scripts must run in order, because later ones load the earlier ones' outputs from the same folder:
- e1 saves `pilot34.pkl`, which holds the teacher logits;
- e2 saves `pilot34b.pkl`;
- e3 saves `pilot34c.pkl`, which holds the nested out-of-fold teacher logits.

## Scripts

| Script | Question | Outcome |
|---|---|---|
| e1 | How good is a teacher on G, on E+G? First student variants | Teacher on G 0.832 balanced acc. (E+G 0.808). First students no better than the baseline |
| e2 | Students with the prior shift in the soft loss; post-hoc τ | 0.695–0.697 vs baseline 0.691. Post-hoc τ inconsistent |
| e3 | G-prediction head and labelled-row distillation | All three channels 0.707 (+0.016, 1 seed) |
| e4 | Confirm the all-channel student over 3 seeds | 0.708 vs 0.682, +0.026 (se 0.010), 12/15 runs |
| e5 | Student settings sweep (1 seed) | Mostly noise. β = 3 and τ = 1.5 worth confirming |
| e6 | **Self-training on test rows. Experiment only, NOT used in any submission** | Worse: −0.007 to −0.012 |
| e7 | Confirm β = 3, τ = 1.5 over 3 seeds | β = 3: +0.005, no cost. τ = 1.5: +0.004 balanced but −0.016 accuracy, −0.046 far group |
| e8 | How much of G can E predict? Classify through predicted G | R² ≈ 0.47 overall, ≈ 0.05 in the far group. Through predicted G: 0.657–0.667, below the student |

**Code version caveat for e1:** e1's student rows ran with the first version of the soft loss, which had no prior
shift. With the current `ie643/distill.py` the default is `adjust=True`. To reproduce those rows, pass
`StudentConfig(..., adjust=False)`; variant 1 in `phase3_distillation.ipynb` does exactly that.
