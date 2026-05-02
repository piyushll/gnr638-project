"""
Download Qwen2.5-VL weights to ./models/<model-name>/.

Invoked by setup.bash on the grader (with internet ON during the setup phase).

    python download_model.py                       # default: 32b-awq (~20 GB)
    python download_model.py --variant 32b-awq     # Qwen2.5-VL-32B-Instruct-AWQ (primary)
    python download_model.py --variant 3b-awq      # Qwen2.5-VL-3B-Instruct-AWQ (fallback)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

VARIANTS = {
    # PRIMARY: Qwen2.5-VL-32B-Instruct-AWQ — Alibaba's OFFICIAL AWQ quant.
    # No official Qwen3-VL-32B AWQ exists; the official Qwen3-VL-32B-FP8 from
    # the Qwen org is vLLM/SGLang only and not loadable via transformers.
    # We deliberately do not use community-quantized Qwen3-VL AWQs to keep the
    # weight provenance clean and the load path battle-tested.
    "32b-awq": {
        "repo_id": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        "local_name": "Qwen2.5-VL-32B-Instruct-AWQ",
        "approx_gb": 20,
    },
    # Small last-resort fallback inside setup.bash — if the 32B-AWQ load fails
    # on the grader for any reason, inference.py auto-falls-through to this
    # 3B-AWQ which loads cleanly on minimal VRAM.
    "3b-awq": {
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        "local_name": "Qwen2.5-VL-3B-Instruct-AWQ",
        "approx_gb": 4,
    },
}

ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.py",
    "*.model",
    "*.safetensors",
    "tokenizer*",
    "preprocessor*",
    "processor*",
    "chat_template*",
    "generation_config*",
    "merges*",
    "vocab*",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default="32b-awq",
        help="Which checkpoint to pull.",
    )
    p.add_argument(
        "--out-dir",
        default="./models",
        help="Parent directory for the downloaded snapshot.",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Optional commit hash or branch; defaults to main.",
    )
    args = p.parse_args()

    spec = VARIANTS[args.variant]
    out_root = Path(args.out_dir).resolve()
    target = out_root / spec["local_name"]
    target.mkdir(parents=True, exist_ok=True)

    print(
        f"[download] {spec['repo_id']} -> {target} (~{spec['approx_gb']} GB)",
        file=sys.stderr,
    )
    # huggingface_hub copies into local_dir (no symlinks), so the folder is
    # self-contained and inference.py can load it directly via from_pretrained.
    snapshot_download(
        repo_id=spec["repo_id"],
        revision=args.revision,
        local_dir=str(target),
        allow_patterns=ALLOW_PATTERNS,
        max_workers=8,
        # HF hub uses HF_TOKEN / HUGGINGFACE_HUB_TOKEN env vars if set.
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
    )
    print(f"[download] done -> {target}", file=sys.stderr)
    print(str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
