"""Train the final student ensemble and write a submission (run from the repository root).

    python scripts/train_final.py --beta 3 --tau 1.5 --name final_beta3_tau1.5

Fully inductive: teachers and students are trained on Train_Data.csv + Supplementry_Data.csv only.
The test rows are used for prediction only, never for training.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ie643 import cv, data, distill, mlp  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, default=1.0, help="weight of the G-prediction head")
parser.add_argument("--tau", type=float, default=1.0, help="logit-adjustment strength")
parser.add_argument("--teacher-seeds", type=int, default=3)
parser.add_argument("--student-seeds", type=int, default=10)
parser.add_argument("--name", required=True, help="name for the checkpoint folder and the submission file")
args = parser.parse_args()

cfg = mlp.Config(epochs=40, dropout=0.5, input_dropout=0.2, weight_decay=0.05, loss="logit_adjusted", tau=args.tau)
scfg = distill.StudentConfig(lam=2.0, T=2.0, lam_lab=1.0, beta=args.beta)
teacher_seeds, student_seeds = tuple(range(args.teacher_seeds)), tuple(range(args.student_seeds))

d = data.load()
groups, fold = cv.load_folds(d.train_id)
t0 = time.time()
students, _ = distill.fit_final(d.E, d.G, d.y, fold, cfg, scfg, teacher_seeds=teacher_seeds, student_seeds=student_seeds)
proba = mlp.ensemble_proba(students, d.F)
pred = proba.argmax(1)
print(f"trained {len(students)} students in {time.time() - t0:.0f} s")

ckpt_dir = Path("checkpoints") / args.name
ckpt_dir.mkdir(parents=True, exist_ok=True)
for s, m in zip(student_seeds, students):
    torch.save(m.state_dict(), ckpt_dir / f"student_seed{s}.pt")
np.save(ckpt_dir / "test_proba.npy", proba)
(ckpt_dir / "config.json").write_text(json.dumps({"mlp": asdict(cfg), "student": asdict(scfg),
                                                  "teacher_seeds": teacher_seeds, "student_seeds": student_seeds}, indent=1))

sub = pd.DataFrame({"ID": d.test_id.astype(int), "Label": pred.astype(int)})
assert len(sub) == 490 and sub["ID"].is_unique and sub["Label"].between(0, data.N_CLASSES - 1).all()
out = Path("submissions") / f"{args.name}.csv"
out.parent.mkdir(exist_ok=True)
sub.to_csv(out, index=False)
print("wrote", out, "| checkpoints in", ckpt_dir)
print("predicted class mix:", np.round(np.bincount(pred, minlength=data.N_CLASSES) / len(pred), 3))
