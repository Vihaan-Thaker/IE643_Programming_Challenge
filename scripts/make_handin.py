"""Assemble the folder for the Google Drive hand-in (run from the repository root):

    python scripts/make_handin.py --roll 24B1234

Creates handin/<ROLL>_IE643_Programming_Challenge/ (and a .zip of it) with:
- every file in git (tracked, or new and not ignored): code, notebooks with outputs, README.md, docs/REPORT.md, figures, experiment log, folds;
- the final model weights: checkpoints/phase3_student/;
- the final submission: submissions/phase3_student_all_channels.csv.
The competition data is not included.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

FINAL_CHECKPOINTS = Path("checkpoints/phase3_student")
FINAL_SUBMISSION = Path("submissions/phase3_student_all_channels.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--roll", required=True, help="your roll number")
args = parser.parse_args()

root = Path(__file__).resolve().parent.parent
out = root / "handin" / f"{args.roll}_IE643_Programming_Challenge"
if out.exists():
    shutil.rmtree(out)

tracked = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, capture_output=True, text=True, check=True).stdout.split()
for rel in tracked + [str(FINAL_SUBMISSION)]:
    src = root / rel
    if not src.is_file():
        raise SystemExit(f"missing: {rel}")
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out / rel)
if not (root / FINAL_CHECKPOINTS).is_dir():
    raise SystemExit(f"missing: {FINAL_CHECKPOINTS}")
shutil.copytree(root / FINAL_CHECKPOINTS, out / FINAL_CHECKPOINTS)

archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
n_files = sum(1 for p in out.rglob("*") if p.is_file())
size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
print(f"{out.relative_to(root)}: {n_files} files, {size_mb:.1f} MB")
print(f"zip: {Path(archive).relative_to(root)}")
