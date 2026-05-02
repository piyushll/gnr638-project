# GNR638 — Project 2: Visual MCQ Solver

Zero-shot visual multiple-choice solver for deep-learning question pages
rendered as PNG images. Built on **Qwen2.5-VL-32B-Instruct-AWQ** with
self-consistency voting and a strict output guard (option always in
`{1,2,3,4,5}`, never the −1-penalty hallucination class).

## 1. Repo layout

```
.
├── setup.bash         # provisioning script (the ONLY file in the submitted zip)
├── inference.py       # grader entry point: --test_dir <abs_path>
├── solver.py          # Qwen2.5-VL inference engine + self-consistency
├── download_model.py  # one-shot HF snapshot fetch into ./models/
├── requirements.txt   # pip deps (installed by setup.bash)
├── README.md          # this file
├── .gitignore         # excludes models/, __pycache__/, dev tools
└── .gitattributes     # forces LF line endings (.bash / .sh / .py / .txt / .md)
```

## 2. Grading workflow (per project guidelines)

The grader executes, in order:

```bash
cd ./your_directory
bash setup.bash                                              # internet ON
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>   # internet OFF
python <grading_script> --submission_file submission.csv
conda remove --name gnr_project_env --all -y
```

`setup.bash`:
- Clones this public GitHub repo (https://github.com/piyushll/gnr638-project) into the cwd
- Creates the conda env `gnr_project_env` (Python 3.11)
- `pip install -r requirements.txt` into that env
- Downloads `Qwen2.5-VL-32B-Instruct-AWQ` (~22 GB primary) and
  `Qwen2.5-VL-3B-Instruct-AWQ` (~3 GB fallback) into `./models/`
- Total disk usage: ~25 GB (well under the 50 GB cap)

`inference.py`:
- Reads `<test_dir>/test.csv` and `<test_dir>/images/`
- Writes `submission.csv` in the script's own directory (NOT inside `test_dir`)
- Sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`
  before any HuggingFace import — guaranteed offline operation
- Tries the 32B-AWQ first; on any load failure, falls through to the 3B-AWQ
- On successful load, runs 3-sample self-consistency voting per image
- On total catastrophic failure, leaves a pre-written all-5s
  `submission.csv` on disk and exits 0 (no crash, no missing file)

## 3. submission.csv schema

```
id,image_name,option
image_1,image_1,3
image_2,image_2,5
...
```

- `id` is identical to `image_name` (per TA clarification)
- `option ∈ {1,2,3,4}` → 1=A, 2=B, 3=C, 4=D
- `option = 5` → unanswered (no penalty). Emitted only when no letter is
  parsed from any sample, or majority vote is tied
- Any other value would be scored as hallucinated (−1). A final
  `.apply(lambda v: int(v) if int(v) in {1..5} else 5)` clamp makes that
  structurally impossible

Score: `correct − 0.25 × wrong − 1 × hallucinated`.

## 4. Hard-requirement compliance summary

| Requirement | How we satisfy it |
|---|---|
| No internet during inference | HF offline env vars set before any import (`inference.py:17–19`) |
| `inference.py --test_dir <abs>` | Required CLI arg defined in `inference.py:62` |
| `submission.csv` in repo dir, not test dir | `out_csv = REPO_DIR / "submission.csv"` (`inference.py:76`) |
| Conda env `gnr_project_env`, Python 3.11 | Hardcoded in `setup.bash:21–22` |
| 48 GB VRAM cap | 32B-AWQ uses ~22 GB; 3-sample KV cache + activations bring peak to ~28 GB |
| 16 GB RAM cap | `low_cpu_mem_usage=True` + `device_map="auto"` stream weights to GPU layer-by-layer; peak ~5–8 GB CPU RSS |
| 50 GB disk cap | ~22 GB (32B-AWQ) + ~3 GB (3B-AWQ) + ~5 GB (conda env) ≈ 30 GB |
| 1-hour inference cap | ~25–45 min estimated for 50 images with 3-sample self-consistency |
| Never crash → 0 marks | Pre-load fallback CSV + multi-tier model fallback chain + per-image try/except |
| LF line endings (Linux) | Enforced via `.gitattributes` |

## 5. Design notes

- **Why a VLM, not OCR→LLM.** The pages contain LaTeX math and PyTorch code
  blocks. OCR on LaTeX / monospace is fragile — one botched subscript or
  mis-tokenized identifier corrupts downstream reasoning. A single VLM pass
  sees the rendered pixels end-to-end.
- **Why Qwen2.5-VL-32B-Instruct-AWQ.** Strongest open-weight VLM at math +
  code + screenshot reasoning that fits in 48 GB VRAM with margin. Uses
  Alibaba's official AWQ quantization (no community weights in the chain).
- **Self-consistency.** 3 samples at T=0.6, top_p=0.9, majority vote.
  Batched via `num_return_sequences` so the image prefill (the bulk of VLM
  latency) is amortized across samples — effective slowdown vs greedy is
  ~1.3×, not 3×.
- **Never hallucinate the output.** Parser extracts `ANSWER: <A|B|C|D>`
  with regex fallback to a trailing letter; any miss → 5. A final
  DataFrame-level clamp enforces `{1..5}`. The −1 penalty is 4× the
  per-wrong penalty — avoiding it is the single most important guard.
- **Always answer when a letter is parsed.** Random-guess EV over 4 options
  is `0.25·(+1) + 0.75·(−0.25) = +0.0625`. Skipping a parsed answer is
  net-negative; we only emit 5 on tied votes or total parse failure.
- **Compatibility shim.** `autoawq 0.2.x` imports `PytorchGELUTanh` from
  `transformers.activations`, renamed away in `transformers≥4.52`.
  `solver.py` installs a no-op shim before model load, preventing the
  cascade from breaking the load path.

## 6. Citations

- Bai et al., *Qwen2.5-VL Technical Report*, 2025. Alibaba Qwen team.
  HuggingFace IDs: `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`,
  `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`.
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in
  Language Models*, ICLR 2023.
- `transformers`, `accelerate`, `huggingface-hub`, `autoawq`, `qwen-vl-utils`,
  `torch`, `pillow`, `pandas`, `numpy` — Apache-2.0 / BSD / MIT licenses.
  See `requirements.txt` for version floors.
