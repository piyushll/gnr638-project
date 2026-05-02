"""
Visual MCQ solver around Qwen2.5-VL.

One model, one image per call. The prompt forces a rigid ANSWER: X footer so
parsing is deterministic. Self-consistency across N samples picks the majority
letter. Anything we can't parse confidently falls back to 5 (skip) — NEVER an
out-of-range value (−1 hallucination penalty is 4× worse than a wrong answer).

Usable as a library (import build_solver) or a CLI:

    python solver.py \
        --model-path ./models/Qwen2.5-VL-7B-Instruct \
        --test-csv  ../sample_test_project_2/test.csv \
        --image-dir ../sample_test_project_2/images \
        --out-csv   ./submission.csv \
        --num-samples 5
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image

# transformers import is deferred to build_solver so `python solver.py --help`
# works without a CUDA env.


def _install_awq_compat_shim() -> None:
    """autoawq 0.2.9 imports `PytorchGELUTanh` from transformers.activations,
    which was renamed away in transformers>=4.52. Transformers' internal AWQ
    loader still imports autoawq, so the cascade breaks model loading even
    though the symbol isn't needed at runtime.

    Stub it in if missing; the fix is a no-op on older transformers.
    """
    try:
        import transformers.activations as _acts
        import torch.nn as _nn
        import torch.nn.functional as _F
    except ImportError:
        return
    if hasattr(_acts, "PytorchGELUTanh"):
        return

    class PytorchGELUTanh(_nn.Module):  # pragma: no cover - trivial shim
        def forward(self, x):
            return _F.gelu(x, approximate="tanh")

    _acts.PytorchGELUTanh = PytorchGELUTanh


# A→1, B→2, C→3, D→4. 5 = skipped (no penalty).
LETTER_TO_OPTION = {"A": 1, "B": 2, "C": 3, "D": 4}
VALID_OUTPUT = {1, 2, 3, 4, 5}

# Capture: "ANSWER: B", "Answer: (C)", "Final answer - D.", etc. Letter only.
_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer|correct\s+option|correct\s+answer)\s*[:\-]?\s*"
    r"[\(\[\{]?\s*([A-D])\b",
    re.IGNORECASE,
)
# Fallback: a standalone letter at the very end of the generation.
_TRAILING_LETTER_RE = re.compile(r"\b([A-D])\b[\s\.\)\]\}]*\s*$")


SYSTEM_PROMPT = (
    "You are an expert deep-learning teaching assistant. You are shown ONE "
    "multiple-choice question from a deep-learning course as a page image. "
    "The page has a question (with possible LaTeX math or PyTorch code) and "
    "four options labeled A, B, C, D. Exactly one option is correct.\n"
    "\n"
    "Work through the problem carefully: identify what is being asked, do any "
    "required calculation or code-tracing, then compare each option on merit "
    "and decide. Do not skip reasoning, and do not just echo the options.\n"
    "\n"
    "End your answer with a single line in exactly this format:\n"
    "ANSWER: X\n"
    "where X is one of A, B, C, D. Nothing may follow that line."
)

USER_INSTRUCTION = (
    "Solve the MCQ shown in the image. Think through the problem, then "
    "finish with the line `ANSWER: <letter>`."
)


@dataclass
class SolverConfig:
    model_path: str
    num_samples: int = 5
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 512
    min_vote_fraction: float = 0.0  # 0 -> always answer if any letter parsed
    dtype: str = "auto"             # "auto" | "bfloat16" | "float16"
    max_image_pixels: int = 1280 * 1280  # cap to keep latency bounded


class QwenVLSolver:
    def __init__(self, cfg: SolverConfig):
        _install_awq_compat_shim()
        # AutoModelForImageTextToText auto-resolves to the correct model class
        # from the snapshot's config.json — works for both
        # Qwen2_5_VLForConditionalGeneration (Qwen2.5-VL) and Qwen3VL... (Qwen3-VL).
        # Falls back to Qwen2_5_VL... explicitly if the auto class isn't
        # registered (very old transformers).
        from transformers import AutoProcessor

        self.cfg = cfg

        dtype_arg: object
        if cfg.dtype == "auto":
            dtype_arg = "auto"
        elif cfg.dtype == "bfloat16":
            dtype_arg = torch.bfloat16
        elif cfg.dtype == "float16":
            dtype_arg = torch.float16
        else:
            raise ValueError(f"unknown dtype: {cfg.dtype}")

        print(f"[solver] loading {cfg.model_path} (dtype={cfg.dtype})", file=sys.stderr)
        # transformers>=4.52 renamed torch_dtype -> dtype. Try the new name first
        # and fall back for older installs.
        load_kwargs = dict(device_map="auto", low_cpu_mem_usage=True)

        try:
            from transformers import AutoModelForImageTextToText
            ModelCls = AutoModelForImageTextToText
        except ImportError:
            # Very old transformers — pin to Qwen2.5-VL class explicitly.
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls  # type: ignore

        try:
            self.model = ModelCls.from_pretrained(
                cfg.model_path, dtype=dtype_arg, **load_kwargs
            )
        except TypeError:
            self.model = ModelCls.from_pretrained(
                cfg.model_path, torch_dtype=dtype_arg, **load_kwargs
            )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(cfg.model_path)
        # Lower the processor's default max pixel budget to cap per-image tokens.
        try:
            self.processor.image_processor.max_pixels = cfg.max_image_pixels
        except AttributeError:
            pass
        print("[solver] model loaded", file=sys.stderr)

    # ---- generation ------------------------------------------------------

    @torch.inference_mode()
    def _generate_samples(self, image: Image.Image, n: int) -> list[str]:
        """Run a single batched generate() returning n samples.

        Image prefill dominates VLM latency, so sharing it across samples via
        num_return_sequences is materially faster than looping. All samples are
        sampled (T>0); for n==1 we fall back to greedy.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_INSTRUCTION},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            pad_token_id=self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id,
        )
        if n <= 1:
            gen_kwargs.update(do_sample=False, num_return_sequences=1)
        else:
            gen_kwargs.update(
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                num_return_sequences=n,
            )

        out = self.model.generate(**inputs, **gen_kwargs)
        # out shape: (n, prompt_len + new_tokens). Strip the echoed prompt.
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out[:, prompt_len:]
        decoded = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [s.strip() for s in decoded]

    # ---- parsing + voting ------------------------------------------------

    @staticmethod
    def parse_letter(text: str) -> str | None:
        """Extract A/B/C/D from a generation. Returns None if none found."""
        m = _ANSWER_RE.search(text)
        if m:
            return m.group(1).upper()
        m = _TRAILING_LETTER_RE.search(text.strip())
        if m:
            return m.group(1).upper()
        return None

    def solve_image(self, image_path: Path) -> tuple[int, list[str]]:
        """Return (option, raw_generations).

        option ∈ {1,2,3,4,5}. 5 means "skip" (no letter parsed OR vote tied
        below min_vote_fraction). Never returns anything else.
        """
        image = Image.open(image_path).convert("RGB")

        n = max(1, self.cfg.num_samples)
        generations = self._generate_samples(image, n)
        votes: Counter[str] = Counter()
        for text in generations:
            letter = self.parse_letter(text)
            if letter in LETTER_TO_OPTION:
                votes[letter] += 1

        if not votes:
            return 5, generations

        (top_letter, top_count), = votes.most_common(1)
        # If a tie for the top count exists, skip (vote is unreliable).
        tied = [l for l, c in votes.items() if c == top_count]
        if len(tied) > 1:
            return 5, generations

        if top_count / n < self.cfg.min_vote_fraction:
            return 5, generations

        option = LETTER_TO_OPTION[top_letter]
        # Paranoia: if anything above ever went wrong, clamp to skip.
        if option not in VALID_OUTPUT:
            return 5, generations
        return option, generations


def build_solver(cfg: SolverConfig) -> QwenVLSolver:
    return QwenVLSolver(cfg)


# ---- CLI -----------------------------------------------------------------

def _iter_rows(test_csv: Path) -> Iterable[tuple[str, str]]:
    df = pd.read_csv(test_csv)
    if "image_name" not in df.columns:
        raise ValueError(f"{test_csv} has no image_name column: {list(df.columns)}")
    for _, row in df.iterrows():
        name = str(row["image_name"])
        image_id = str(row.get("image_id", name))
        yield image_id, name


def run(args: argparse.Namespace) -> int:
    cfg = SolverConfig(
        model_path=args.model_path,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        min_vote_fraction=args.min_vote_fraction,
        dtype=args.dtype,
    )
    solver = build_solver(cfg)

    image_dir = Path(args.image_dir)
    test_csv = Path(args.test_csv)
    out_csv = Path(args.out_csv)

    rows: list[dict[str, object]] = []
    t0 = time.time()
    for i, (image_id, image_name) in enumerate(_iter_rows(test_csv), start=1):
        # Images are expected to be .png but the code handles any Pillow-readable
        # extension for robustness.
        candidates = [
            image_dir / f"{image_name}.png",
            image_dir / image_name,
            image_dir / f"{image_name}.jpg",
            image_dir / f"{image_name}.jpeg",
        ]
        img_path = next((p for p in candidates if p.exists()), None)
        if img_path is None:
            print(f"[solve] {image_name}: missing image -> skip(5)", file=sys.stderr)
            rows.append({"image_name": image_name, "option": 5})
            continue

        option, gens = solver.solve_image(img_path)
        dt = time.time() - t0
        print(
            f"[solve] {i:>3} {image_name}: option={option} "
            f"(elapsed={dt:.1f}s)",
            file=sys.stderr,
        )
        if args.verbose:
            for j, g in enumerate(gens):
                print(f"  sample[{j}]: {g!r}", file=sys.stderr)
        rows.append({"image_name": image_name, "option": option})

    # Guard the output schema one last time.
    out_df = pd.DataFrame(rows, columns=["image_name", "option"])
    out_df["option"] = out_df["option"].apply(lambda v: v if v in VALID_OUTPUT else 5)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"[solve] wrote {out_csv} ({len(out_df)} rows)", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out-csv", default="submission.csv")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--min-vote-fraction",
        type=float,
        default=0.0,
        help="Skip (predict 5) if top vote share < this fraction. Default 0 "
        "(answer whenever any letter is parsed).",
    )
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
