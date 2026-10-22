# IE643 Programming Challenge (2026)

Predict one of 11 labels (0–10) for each test sample. The samples are 768-dimensional feature vectors, and about 90% of
them are "deformed". Only MLPs and CNNs are allowed, and no external data.

**Final submission: a distilled student MLP.**
- A teacher MLP is trained on the 384 supplementary features, which exist only for training rows.
- A student MLP on the 768 main features learns from it, through pseudo-labels, out-of-fold soft labels and an auxiliary head that predicts the supplementary features.
- The submission is an ensemble of 5 students.

| | CV balanced accuracy | CV accuracy | Kaggle public score |
|---|---|---|---|
| Phase 1 baseline: MLP on the main features, logit-adjusted loss | 0.682 | 0.775 | 0.6165 |
| **Final: distilled student (Phase 3)** | **0.708** | **0.792** | **0.6619** |

CV is 5 fixed folds × 3 seeds. The full method, results and alternatives tried are in [`docs/REPORT.md`](docs/REPORT.md),
and every experiment is in [`docs/experiment_log.md`](docs/experiment_log.md).

**Rule compliance**
- **Models:** every model that is trained or used for predictions is an MLP (PyTorch).
- **Data:** no external data.
- **Training rows:** the final model is trained only on `Train_Data.csv` and `Supplementry_Data.csv`. The test rows are used only for prediction.
- **Linear probes:** scikit-learn linear probes appear in the EDA, only as data diagnostics.

## Repository structure

```
├── README.md                     this file
├── requirements.txt              Python dependencies
├── ie643/                        shared code
│   ├── data.py                   loads the CSVs into numpy arrays
│   ├── cv.py                     Phase 0: groups (clean/deformed/far), fixed folds, metrics
│   ├── mlp.py                    Phase 1: MLP, the three losses, cross-validation runner
│   └── distill.py                Phases 3–4: teacher/student distillation, final-model training
├── folds/cv_folds_v1.json        the fixed CV folds used by every experiment
├── scripts/
│   ├── make_folds.py             recreates folds/cv_folds_v1.json
│   └── train_final.py            trains the final ensemble and writes the submission
├── eda_deformed.ipynb            EDA: what "deformed" means, label/test-set analysis
├── phase1_baseline_mlp.ipynb     Phases 0–1: validation protocol, baseline MLP, loss comparison
├── phase3_distillation.ipynb     Phases 3–4: teacher, distillation channels, final student
├── experiments/                  scripts + console logs for the tuning and negative-result experiments
└── docs/
    ├── REPORT.md                 the detailed report
    ├── figures/                  figures used in the report
    └── experiment_log.md         documentation kit: one entry per experiment
```

Not in git (git-ignored):
- `ie-643-2026-programming-challenge/`: the competition data;
- `checkpoints/`: model weights. The final model is in `checkpoints/phase3_student/`;
- `submissions/`: the submission CSVs. The final one is `submissions/phase3_student_all_channels.csv`.

## Setup

Python 3.12. Tested with numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, PyTorch 2.14.1 (CPU) and matplotlib 3.11.2.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Use `source .venv/bin/activate` on Linux/macOS. Put the competition files in `ie-643-2026-programming-challenge/` at the
repository root:

```
ie-643-2026-programming-challenge/
├── Train_Data.csv
├── Test_Data.csv
├── Supplementry_Data.csv
└── Sample_Submission.csv
```

## Reproducing the final submission

From the repository root:

```bash
python scripts/train_final.py --beta 1 --tau 1 --teacher-seeds 3 --student-seeds 5 --name final_phase3
```

The script:
- trains 3 teacher MLPs on the supplementary features of all labelled rows, which label the 792 unlabelled rows;
- computes out-of-fold teacher predictions for the labelled rows, using the fixed folds;
- trains 5 student MLPs on the main features of all 2076 training rows;
- writes `submissions/final_phase3.csv`, plus the weights and config to `checkpoints/final_phase3/`.

It takes about 4 minutes on CPU. On the development machine its predictions match the submitted file exactly
(`submissions/phase3_student_all_channels.csv`).

**Using the saved weights instead of retraining.** Load each `checkpoints/phase3_student/student_seed*.pt` into
`ie643.distill.MultiTaskMLP`, average the softmax outputs and take the argmax:

```python
import json, torch, numpy as np
from ie643 import data, distill, mlp
cfg_all = json.load(open("checkpoints/phase3_student/config.json"))
cfg = mlp.Config(**{**cfg_all["mlp"], "hidden": tuple(cfg_all["mlp"]["hidden"])})
d = data.load()
models = []
for s in cfg_all["student_seeds"]:
    state = torch.load(f"checkpoints/phase3_student/student_seed{s}.pt")
    m = distill.MultiTaskMLP(state["mean"].numpy(), state["std"].numpy(), cfg, g_dim=384)
    m.load_state_dict(state)
    models.append(m)
pred = mlp.ensemble_proba(models, d.F).argmax(1)
```

## Re-running the analysis

Run the notebooks from the repository root with the `.venv` kernel, in this order:

| Notebook | Content | CPU time |
|---|---|---|
| `eda_deformed.ipynb` | Data analysis | < 1 min |
| `phase1_baseline_mlp.ipynb` | Validation protocol, baseline, losses | about 20 min |
| `phase3_distillation.ipynb` | Teacher, distillation, final student | about 30 min |

The scripts in `experiments/` (tuning, the test-row self-training experiment that is *not* used, and the other negative
results) are described in `experiments/README.md`.

## Preparing the Drive hand-in

```bash
python scripts/make_handin.py --roll <your roll number>
```

This creates `handin/<roll>_IE643_Programming_Challenge/` and a `.zip` of it. It contains the code, the notebooks with
outputs, this README, the report and its figures, the experiment log, the folds, the final checkpoints and the final
submission. The competition data is not included.
