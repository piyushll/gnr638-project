# Project 2 — Approach Document

GNR638 Visual Deep-Learning MCQ Solver. This document explains *what* we are
solving, *why* we chose this approach, and *how* the system fits together.
For setup / operations, see `README.md`.

---

## 1. The problem

We are given up to 50 PNG images. Each image is a single-page rendering of a
deep-learning multiple-choice question:

- A question stem (often containing LaTeX math, network diagrams, or PyTorch
  code blocks)
- A header `Options`
- Four answer options labelled `A`, `B`, `C`, `D` — exactly one is correct

For each image, our system must emit one of:

| Output | Meaning | Score per question |
|---|---|---|
| `1` | Predicted A | `+1` if correct, `-0.25` otherwise |
| `2` | Predicted B | same |
| `3` | Predicted C | same |
| `4` | Predicted D | same |
| `5` | Skip (no answer) | `0` |
| anything else | Hallucinated | **`-1`** |

The hallucination penalty (`-1`) is 4× the wrong-answer penalty. **Never
emitting an out-of-range value** is the single most important constraint —
more important than picking correct answers.

Final score = `correct − 0.25 × wrong − 1 × hallucinated`.

The output is a CSV with columns `id, image_name, option` (where `id == image_name`).

---

## 2. The hard constraints

| Constraint | Source | Implication |
|---|---|---|
| **No internet during inference** | Spec | All weights, deps, and code must be local before `inference.py` runs |
| **1-hour cap on inference** | Spec | Every per-image second matters; we can't run a frontier-scale model serially |
| **48 GB VRAM (L40s)** | Spec | Excludes 70B+ models without aggressive offloading; 32B-quantized fits comfortably |
| **16 GB system RAM** | Spec | Must avoid loading full fp16 weights into CPU RAM (use `low_cpu_mem_usage=True`) |
| **Linux + CUDA 12.6** | Spec | Native CUDA wheels (cu124 binaries are forward-compatible to 12.6 driver) |
| **Single `setup.bash` zip** | Spec | All artefacts (code, weights, env) must be reproducible from one bash script that has internet during setup |
| **Conda env name `gnr_project_env`, Python 3.11** | Spec | Hard-coded into `setup.bash` |
| **Auto-graded, no human in the loop** | Spec | Any crash, missing file, or wrong column → 0. Defensive coding required |

---

## 3. The approach in one sentence

**Use a strong open-weight Vision-Language Model (Qwen2.5-VL-32B-AWQ) zero-shot
on each image, with self-consistency voting and a hard output guard that
makes the −1 hallucination penalty structurally impossible.**

No fine-tuning, no synthetic dataset, no ensemble. The TA confirmed pretrained
VLMs without fine-tuning are acceptable; we don't have the time, budget, or
labelled data to do better than a well-prompted 32B model.

---

## 4. Why a VLM, not OCR→LLM?

The natural alternative is a two-stage pipeline:
1. OCR the image to text
2. Feed text + question to an LLM

We rejected this because:

- **LaTeX rendering breaks OCR.** The samples contain expressions like
  `h₁ = ReLU(W₁x + b₁)` rendered as Computer Modern math. Off-the-shelf OCR
  routinely loses subscripts, mistakes `W₁` for `Wi`, drops superscripts.
  One mangled symbol corrupts the entire downstream reasoning.
- **Code blocks need monospace fidelity.** `nn.Linear(in_dim, h1)` vs
  `nn.Linear(in dim, h1)` — a single missed underscore changes meaning.
  General-purpose OCR (Tesseract, PaddleOCR) is unreliable on monospace
  Python source against arbitrary backgrounds.
- **Specialized OCR is heavier than the VLM.** GOT-OCR2.0 (~6B params) +
  any reasoning LLM (~7-30B) is roughly the same parameter budget as a
  single VLM, but with two loading and two inference passes per image.
- **VLMs see the rendered pixels end-to-end.** No information loss from
  text extraction. The model jointly handles math notation, code, layout,
  and reasoning.

The downside of going VLM-only is that vision-language alignment is harder
than pure language modelling — but on this kind of academic-paper-style
content, modern open-weight VLMs (especially Qwen2.5-VL) are strong.

---

## 5. Why Qwen2.5-VL-32B-Instruct-AWQ specifically

We need a model that:
- Handles math notation, PyTorch code, and reasoning chains
- Fits in 48 GB VRAM with headroom for activations + KV cache
- Has accessible pre-trained weights (HuggingFace, no API key gate)

Comparing the realistic options for our hardware budget:

| Model | Params | Quantized size | Fits L40s? | Reasoning quality |
|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct(-AWQ) | 7B | 5–17 GB | ✓ easily | Decent — borderline on hard math |
| **Qwen2.5-VL-32B-Instruct-AWQ** | **32B** | **~22 GB** | **✓ comfortably** | **Strong** ← chosen |
| Qwen2.5-VL-72B-Instruct-AWQ | 72B | ~42 GB | borderline | Strongest, but peak with 3-sample KV cache > 48 GB |
| InternVL2.5-26B(-AWQ) | 26B | ~14 GB | ✓ | Comparable to Qwen2.5-32B; different code path |
| Pixtral-12B / Llama-3.2-11B-Vision | 11–12B | ~7–22 GB | ✓ | Weaker on code/math specifically |

**Qwen2.5-VL-32B-AWQ is the sweet spot.** It is:

- The strongest open-weight VLM for math/code/screenshot reasoning we can run
  comfortably with margin.
- 4-bit AWQ quantized, so weights are ~22 GB on disk and at load — small
  enough that `huggingface_hub.snapshot_download` finishes in ~5–10 minutes
  on a typical cloud GPU's network, leaves ~30 GB of disk under the 50 GB cap.
- Has prebuilt `Qwen2_5_VLForConditionalGeneration` support in
  `transformers>=4.49`, so no custom modelling code.
- Uses the same chat template / processor as the much smaller 3B-AWQ
  variant we use for laptop smoke testing — pipeline correctness transfers.

**Why we did NOT pick the 72B-AWQ:** at 3-sample self-consistency the KV cache
plus visual tokens push peak VRAM past 48 GB. OOM = 0 marks. The marginal
accuracy gain isn't worth the failure risk.

---

## 6. Self-consistency: 3 samples, majority vote

Instead of a single greedy generation, we sample **N = 3** generations per
image at temperature 0.6, top-p 0.9, then majority-vote the parsed letter.

### Why self-consistency

When a model is uncertain, individual samples disagree; when it's confident,
they cluster. Voting filters out flukes (a sampled token that happens to lead
the model down a wrong path) without needing logprob calibration.

This is the [Wang et al., ICLR 2023](https://arxiv.org/abs/2203.11171) trick
and consistently boosts CoT reasoning by 5-15 points on math tasks.

### Why N = 3 specifically

- N = 1 (greedy) loses the variance-reduction benefit.
- N = 5+ has diminishing returns and risks blowing the 1-hour budget.
- **N = 3 is the largest value where 50 images × per-image latency comfortably
  fits inside one hour on the L40s** (estimated ~25–45 minutes).

### Why batching doesn't make this 3× slower

The image prefill (vision encoder forward + LM prefill on the visual tokens)
dominates VLM latency. We use `num_return_sequences=3` so one prefill is
shared across all three sampled generations — only the autoregressive decode
is repeated. Effective slowdown vs greedy is ~1.3× rather than 3×.

---

## 7. The output guard: −1 is structurally impossible

Three layers prevent ever emitting a hallucinated value:

1. **Robust parser.** `solver.py` first looks for an explicit
   `ANSWER: <A|B|C|D>` footer (the prompt forces this format), then falls back
   to a trailing standalone letter. If neither matches, it returns `None`.
2. **Vote logic.** If no sample produces a parseable letter, or if the top
   vote is tied between two letters, we emit `5` (skip) instead of guessing.
3. **DataFrame-level clamp.** Before writing the CSV, every `option` value
   passes through `lambda v: int(v) if int(v) in {1,2,3,4,5} else 5`. Even if
   a bug somewhere upstream produced a `7` or `-1`, it gets coerced to `5`.

Result: the −1 penalty cannot occur on any well-formed input.

---

## 8. The skip policy

Skipping (output = 5) earns 0 points. Random guessing on 4 options has
expected value `0.25·(+1) + 0.75·(−0.25) = +0.0625` per question. So:

> **As long as the model is better than random, every parsed answer is
> positive-EV.** Only skip when there's no signal.

Concretely, we skip only when:
- All N samples failed to produce any A/B/C/D letter, OR
- The top vote is tied between two or more letters (genuinely ambiguous)

Otherwise we always emit the majority letter, even if the vote is 1-1-1
across three different letters (in which case we'd emit the first one — but
this is rare with N=3).

This policy is conservative on hallucination but aggressive on attempting.
The math says aggressive attempting wins as long as accuracy exceeds 25%,
which a well-prompted 32B VLM exceeds by a wide margin.

---

## 9. Defensive design for the auto-grader

The grader is fully automated. There's no TA who will see "almost worked"
and award partial credit. Every failure mode = 0 marks. We added several
defensive layers:

### Pre-load fallback CSV
**Before** loading the model (which is the most failure-prone step),
`inference.py` writes a placeholder `submission.csv` of all-5s. If the model
fails to load, the grader at least sees a valid CSV with 0 score (no
hallucinations). The successful run overwrites it at the end.

### Per-image try/except
A single CUDA OOM or transformers internal error on image #17 must not
crash the whole run. `inference.py` catches per-image exceptions and emits
`5` for that image, then continues.

### Idempotent setup.bash
- `cp -rf` instead of `cp -r` — so re-running setup.bash after a partial
  failure doesn't error on existing files.
- `conda env remove ... || true` before create — so a stale env doesn't
  block re-provisioning.
- `command -v conda` and `command -v git` checks at the top — fail fast
  with a clear error rather than failing mysteriously later.

### Strict offline mode
`inference.py` sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1` *before* any HuggingFace import. The transformers
library will refuse network access even if the grader's box happens to
have internet — guaranteeing reproducibility.

### Output schema enforcement
The CSV is built explicitly with `columns=["id", "image_name", "option"]`
and `id == image_name`. Even if a future change to the row-building logic
forgets to set `id`, pandas would emit a column-mismatch error before the
CSV write — easy to catch in testing.

---

## 10. The complete pipeline, file by file

```
your_directory/                  ← grader cd's here, runs bash setup.bash
├── setup.bash                   (the only file in the submitted zip)
├── README.md                    operational docs
├── APPROACH.md                  this file
├── requirements.txt             pip dependency floors
├── environment.yml              conda env spec (name=gnr_project_env, py=3.11)
├── download_model.py            HF snapshot fetcher
├── inference.py                 grader entry point: --test_dir <abs>
├── solver.py                    model interface + self-consistency engine
├── score.py                     local scoring against answers.csv (dev tool)
├── make_hard_dataset.py         renders dataset_hard/ for local validation
├── dataset_hard/                50 self-authored MCQs for accuracy testing
│   ├── images/image_*.png
│   ├── test.csv
│   ├── sample_submission.csv
│   └── answers.csv              ground truth (NOT shipped to grader)
└── models/                      populated by setup.bash, gitignored
    └── Qwen2.5-VL-32B-Instruct-AWQ/
        ├── config.json
        ├── model.safetensors
        ├── ... (tokenizer, processor, etc.)
```

### Grading sequence (literal commands the TA will run)

```bash
cd ./your_directory
bash setup.bash                                              # setup phase, internet ON
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>   # inference, internet OFF
python <grading_script> --submission_file submission.csv
conda remove --name gnr_project_env --all -y
```

### What `setup.bash` does (internet ON, time uncapped)
1. Sanity-check `conda` and `git` are on PATH.
2. `git clone --depth=1` the public GitHub repo into a temp dir.
3. Copy repo contents into `your_directory/`, preserving the original
   `setup.bash` (the grader's, not the cloned one).
4. `conda create -n gnr_project_env python=3.11`.
5. `pip install -r requirements.txt` inside that env (torch, transformers,
   accelerate, autoawq, qwen-vl-utils, pillow, pandas, etc.).
6. `python download_model.py --variant 32b-awq` to fetch ~22 GB of weights
   into `./models/Qwen2.5-VL-32B-Instruct-AWQ/`.
7. Final sanity check: `inference.py` exists, `models/` directory exists.

### What `inference.py` does (internet OFF, must finish in 1 hour)
1. Set HuggingFace offline env vars before any HF import.
2. Parse `--test_dir`.
3. Pre-write a fallback `submission.csv` of all-5s.
4. Find the model directory under `./models/` (priority: 32B-AWQ → 7B → 3B-AWQ).
5. Load Qwen2.5-VL via `Qwen2_5_VLForConditionalGeneration.from_pretrained`
   with `device_map="auto"`, `low_cpu_mem_usage=True`, `dtype=fp16`.
6. For each row in `test.csv`:
   - Find the corresponding image file under `<test_dir>/images/`.
   - Generate 3 samples with the chat-template prompt (system + image + instruction).
   - Parse `ANSWER: <letter>` from each generation.
   - Majority-vote → option in `{1,2,3,4}` or `5` (skip).
7. Build and write the final 3-column CSV (overwrites the fallback).

### What `solver.py` provides
- `SolverConfig`: model path, sample count, generation hyperparameters
- `QwenVLSolver`: model loader + image-to-option pipeline
- `solve_image(path) -> (option, raw_generations)`: the single-image entry
- Compatibility shim for `autoawq 0.2.x` × `transformers≥4.52` —
  `PytorchGELUTanh` was renamed away in newer transformers but autoawq
  still imports it; we stub it back in as a no-op before loading.

---

## 11. Validation strategy

We can't run the grader's evaluation; we can only get progressively closer
to it.

| Tier | What it validates | Cost | Fidelity |
|---|---|---|---|
| **Tier 1** — `python -c "import ast; ast.parse(...)"` + `bash -n` | Files parse | 0 | Catches typos only |
| **Tier 2** — Local `inference.py` against 3B-AWQ on laptop GPU | Pipeline works end-to-end with a real (smaller) model | 0 | Wrong accuracy, right shape |
| **Tier 3** — RunPod A6000 (48 GB) running setup.bash + inference + score on `dataset_hard/` | Full grader simulation | ~$0.50 | Closest to real grader |

### The hard dataset
We hand-authored 50 MCQs (`make_hard_dataset.py` + `dataset_hard/`) covering
CNN shape arithmetic, transformer attention, PyTorch code tracing, backprop,
optimizers, losses, activations, BatchNorm, init, regularization, and
architecture trivia. Answer positions are balanced (12-13 each across A/B/C/D)
so a single-letter guesser caps at ~7.5% expected score.

`score.py` evaluates `submission.csv` against `dataset_hard/answers.csv` using
the exact `+1 / −0.25 / 0 / −1` formula the grader uses, giving a real
accuracy number (not just "the script ran").

---

## 12. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `setup.bash` REPO_URL placeholder not edited | High (manual step) | Pre-submission checklist in README |
| GitHub repo private at grading time | Medium (manual step) | Same |
| `pip install autoawq` fails on grader | Low | Fallback to 7B-fp16 by changing `MODEL_VARIANT` |
| Model download timeout | Low | snapshot_download has built-in retry; 50 GB disk margin tolerates partial caches |
| CUDA OOM during inference | Low | `num_samples=3` + 1280×1280 image cap is well within 48 GB. Per-image try/except catches anyway |
| Single bad test image | Low | Per-image try/except → emit 5, continue |
| `conda activate` fails in grader's shell | Very low | Spec mandates the grader runs `conda activate`; that's a precondition we can assume |
| Accuracy lower than expected | Medium | Tested locally on `dataset_hard/`; hard floor is ~7.5% from the skip policy alone |
| Inference exceeds 1 hour | Low | ~25–45 min estimated; if real-world is slower, drop `num_samples` to 1 (~2× speedup) |

---

## 13. Expected outcome

| Metric | Expected | Limit |
|---|---|---|
| Inference runtime, 50 images | 25–45 min | 60 min cap |
| Peak VRAM | ~30 GB | 48 GB cap |
| Peak system RAM | ~5–8 GB | 16 GB cap |
| Disk usage (env + weights + repo) | ~30 GB | 50 GB cap |
| Hallucinated answers (option ∉ {1..5}) | 0 | structurally impossible |
| Skip rate (option = 5) | 5–15% | depends on model confidence |

Accuracy on the actual hidden test set is impossible to predict precisely.
The 32B-AWQ on similar academic-MCQ benchmarks typically lands in the
65–80% range with self-consistency. We do not need to be perfect — we need
to be reliably better than chance with zero hallucinations.

---

## 14. What we explicitly did NOT do

- **No fine-tuning.** The TA confirmed it's not required and we have neither
  labelled training data nor compute budget for it within the deadline.
- **No model ensemble.** Loading two VLMs simultaneously risks OOM and
  doubles per-image latency for marginal accuracy gain.
- **No tool use / sandboxed code execution.** Several questions (e.g.,
  "what is the output of this PyTorch snippet?") could be answered by
  parsing and executing the code. High potential value, high implementation
  risk; not pursued under deadline pressure.
- **No OCR pre-pass.** Discussed in §4. Single-VLM is cleaner.
- **No retrieval-augmented prompting.** No external knowledge base allowed
  during inference; the model must reason from its parameters alone.
- **No chain-of-verification / self-critique loops.** Would multiply
  per-image latency by 2-3× without strong evidence of accuracy gain at
  this scale.

---

## 15. Citations

- Bai et al., *Qwen2.5-VL Technical Report*, 2025. Alibaba Qwen team. HuggingFace IDs: `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`, `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`.
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, ICLR 2023. <https://arxiv.org/abs/2203.11171>
- `transformers`, `accelerate`, `huggingface-hub`, `autoawq`, `qwen-vl-utils`, `pillow`, `pandas`, `numpy`, `torch` — Apache-2.0 / BSD / MIT licenses. See `requirements.txt` for version floors.
