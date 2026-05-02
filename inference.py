"""
Grader entry point.

    python inference.py --test_dir <absolute_path_to_test_dir>

Reads <test_dir>/test.csv and <test_dir>/images/, writes submission.csv next
to this script (NOT inside test_dir). Runs fully offline — no network calls
to HuggingFace.

submission.csv columns: id,image_name,option   (id == image_name)
option ∈ {1,2,3,4,5}: 1=A 2=B 3=C 4=D, 5 = skipped.
"""
from __future__ import annotations

# Force HF stack into offline mode BEFORE any transformers/hub imports.
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from solver import SolverConfig, VALID_OUTPUT, build_solver

REPO_DIR = Path(__file__).resolve().parent

# Best → worst. inference.py iterates this list and uses the first model whose
# folder is populated AND whose `from_pretrained` call succeeds. A load failure
# on Qwen3-VL (e.g., autoawq edge case, missing class in older transformers)
# auto-falls-through to the next entry — guaranteeing we always end up with
# *some* working model as long as setup.bash downloaded both primary + fallback.
MODEL_PRIORITY = (
    "Qwen2.5-VL-32B-Instruct-AWQ",     # PRIMARY — Alibaba official AWQ
    "Qwen2.5-VL-7B-Instruct",          # if user has it locally (no autoawq dep)
    "Qwen2.5-VL-3B-Instruct-AWQ",      # FALLBACK — small, guaranteed to load
)


def iter_candidate_model_dirs() -> "list[Path]":
    base = REPO_DIR / "models"
    found = []
    for name in MODEL_PRIORITY:
        candidate = base / name
        if (candidate / "config.json").exists():
            found.append(candidate)
    if not found:
        raise FileNotFoundError(
            f"no usable model directory under {base}; expected one of "
            f"{MODEL_PRIORITY} with config.json present. setup.bash should have "
            "downloaded these."
        )
    return found


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--test_dir",
        required=True,
        help="absolute path to the test directory (contains test.csv + images/)",
    )
    args = p.parse_args()

    test_dir = Path(args.test_dir).resolve()
    test_csv = test_dir / "test.csv"
    image_dir = test_dir / "images"
    if not test_csv.exists():
        raise FileNotFoundError(f"missing {test_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"missing {image_dir}")

    out_csv = REPO_DIR / "submission.csv"

    # Pre-load fallback: write an all-5s submission.csv up front so even if the
    # model load below crashes, a valid (zero-points-but-zero-penalty) CSV
    # still exists. The successful run overwrites it at the end.
    df_pre = pd.read_csv(test_csv)
    if "image_name" in df_pre.columns:
        fallback = pd.DataFrame({
            "id": df_pre["image_name"].astype(str),
            "image_name": df_pre["image_name"].astype(str),
            "option": 5,
        })
        fallback.to_csv(out_csv, index=False)
        print(f"[infer] wrote fallback {out_csv} (all-5s, will overwrite on success)", file=sys.stderr)

    # Try each model in priority order; fall through on per-model load failure.
    # This is the second safety net (the first being the pre-load fallback CSV
    # already written above). If ALL models fail to load, the all-5s CSV stays
    # on disk and we exit gracefully — no crash, no -1 hallucinations.
    candidates = iter_candidate_model_dirs()
    print(f"[infer] candidate model dirs (in priority order): "
          f"{[c.name for c in candidates]}", file=sys.stderr)

    solver = None
    last_err: BaseException | None = None
    for model_path in candidates:
        cfg = SolverConfig(
            model_path=str(model_path),
            num_samples=3,
            # AWQ checkpoints want fp16 on CUDA; non-quantized models can stay auto.
            dtype="float16" if "AWQ" in model_path.name else "auto",
        )
        try:
            print(f"[infer] attempting to load: {model_path}", file=sys.stderr)
            solver = build_solver(cfg)
            print(f"[infer] loaded: {model_path}", file=sys.stderr)
            break
        except Exception as exc:  # noqa: BLE001 - we want to catch ALL load failures
            print(f"[infer] LOAD FAILED for {model_path.name}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            last_err = exc
            continue

    if solver is None:
        print(f"[infer] ERROR: all candidate models failed to load. "
              f"Last error: {last_err}. Keeping the all-5s fallback CSV.",
              file=sys.stderr)
        # The pre-load fallback CSV is already on disk. Exit 0 so the grader's
        # next command (the grading script) still runs against a valid CSV.
        return 0

    df = pd.read_csv(test_csv)
    if "image_name" not in df.columns:
        raise ValueError(
            f"{test_csv} missing image_name column: {list(df.columns)}"
        )

    rows: list[dict] = []
    t0 = time.time()
    for i, row in df.iterrows():
        name = str(row["image_name"])
        candidates = [
            image_dir / f"{name}.png",
            image_dir / name,
            image_dir / f"{name}.jpg",
            image_dir / f"{name}.jpeg",
        ]
        img_path = next((p for p in candidates if p.exists()), None)
        if img_path is None:
            print(f"[infer] {i + 1} {name}: image missing -> 5", file=sys.stderr)
            rows.append({"id": name, "image_name": name, "option": 5})
            continue
        # Defensive: a per-image failure (CUDA OOM, broken image, transformers
        # internal error, etc.) must NOT kill the whole run. We degrade to skip
        # so the submission.csv is still produced — partial output is salvageable,
        # no output is a guaranteed 0.
        try:
            option, _ = solver.solve_image(img_path)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            print(
                f"[infer] {i + 1} {name}: solver error ({type(exc).__name__}: "
                f"{exc}) -> 5",
                file=sys.stderr,
            )
            option = 5
        print(
            f"[infer] {i + 1:>3} {name}: option={option} "
            f"(elapsed={time.time() - t0:.1f}s)",
            file=sys.stderr,
        )
        rows.append({"id": name, "image_name": name, "option": option})

    out_df = pd.DataFrame(rows, columns=["id", "image_name", "option"])
    # Output guard: clamp anything outside {1..5} to 5 (skip), which avoids
    # the −1 hallucination penalty.
    out_df["option"] = out_df["option"].apply(
        lambda v: int(v) if int(v) in VALID_OUTPUT else 5
    )
    out_df.to_csv(out_csv, index=False)
    print(f"[infer] wrote {out_csv} ({len(out_df)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
