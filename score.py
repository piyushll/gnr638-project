"""
score.py — evaluate submission.csv against dataset_hard/answers.csv.

    python score.py
    python score.py --submission /path/to/submission.csv --answers /path/to/answers.csv

Scoring (per the project brief):
    +1   correct
    -0.25 wrong (option in {1..4} but != answer)
     0   skipped (option == 5)
    -1   hallucinated (option not in {1..5}) — should be impossible with our
         output guard, but we count it for completeness.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent


def load_options(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, int] = {}
    for r in rows:
        # accept either "image_name" or "id" as key, prefer image_name
        key = r.get("image_name") or r.get("id")
        if key is None:
            raise ValueError(f"{path}: row missing image_name/id: {r}")
        try:
            opt = int(r["option"])
        except (KeyError, ValueError):
            opt = -999  # forces "hallucinated" classification
        out[key] = opt
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--submission", default=str(REPO_DIR / "submission.csv"))
    p.add_argument("--answers", default=str(REPO_DIR / "dataset_hard" / "answers.csv"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    sub = load_options(Path(args.submission))
    ans = load_options(Path(args.answers))

    correct = wrong = skipped = halluc = missing = 0
    per_q = []

    for name, true_opt in ans.items():
        if name not in sub:
            missing += 1
            per_q.append((name, true_opt, None, "MISSING"))
            continue
        pred = sub[name]
        if pred == true_opt:
            correct += 1
            per_q.append((name, true_opt, pred, "OK"))
        elif pred == 5:
            skipped += 1
            per_q.append((name, true_opt, pred, "SKIP"))
        elif pred in {1, 2, 3, 4}:
            wrong += 1
            per_q.append((name, true_opt, pred, "WRONG"))
        else:
            halluc += 1
            per_q.append((name, true_opt, pred, "HALLUC"))

    score = correct * 1.0 + wrong * -0.25 + skipped * 0.0 + halluc * -1.0
    n = len(ans)
    max_possible = n * 1.0
    attempted = correct + wrong

    print(f"questions     : {n}")
    print(f"correct       : {correct}")
    print(f"wrong         : {wrong}")
    print(f"skipped       : {skipped}")
    print(f"hallucinated  : {halluc}")
    if missing:
        print(f"missing       : {missing}  (in answers.csv but not submission.csv)")
    print(f"raw score     : {score:+.2f}  (max {max_possible:.0f})")
    if attempted:
        print(f"attempt acc   : {correct}/{attempted} = {100 * correct / attempted:.1f}%")
    print(f"overall acc   : {correct}/{n} = {100 * correct / n:.1f}%")

    if args.verbose:
        print("\nper-question:")
        for name, t, pred, status in per_q:
            print(f"  {name:>12}  truth={t}  pred={pred}  {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
