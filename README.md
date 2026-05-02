# GNR638 — Project 2: Visual MCQ Solver

Zero-shot visual multiple-choice solver for deep-learning question pages
rendered as PNG images. Built on **Qwen2.5-VL** with self-consistency voting
and a strict output guard (option always in `{1,2,3,4,5}`, never the
−1-penalty hallucination class).

## 1. Repo layout

```
.
├── setup.bash         # provisioning script (the ONLY file in the submitted zip)
├── inference.py       # grader entry point: --test_dir <abs>
├── solver.py          # Qwen2.5-VL inference + self-consistency
├── download_model.py  # one-shot HF snapshot fetch into ./models/
├── requirements.txt   # pip deps (installed by setup.bash)
├── environment.yml    # conda env spec (python=3.11, name=gnr_project_env)
├── .gitignore         # excludes models/, __pycache__/, etc.
└── .gitattributes     # forces LF line endings for .bash / .sh / .py
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

`setup.bash` clones this repo into `./your_directory`, creates the conda
env `gnr_project_env` (python 3.11), `pip install -r requirements.txt`,
and downloads the Qwen2.5-VL-32B-Instruct-AWQ weights (~20 GB) into
`./models/`. `inference.py` reads `<test_dir>/test.csv` + `<test_dir>/images/`
and writes `submission.csv` next to itself — **not** inside `test_dir`.

## 3. submission.csv schema

```
id,image_name,option
image_1,image_1,3
image_2,image_2,5
...
```

- `id` is identical to `image_name`.
- `option ∈ {1,2,3,4}` → 1=A, 2=B, 3=C, 4=D.
- `option = 5` → unanswered (no penalty). Emitted only when no letter is
  parsed from any sample, or majority vote is tied.
- Any other value would be scored as hallucinated (−1). A final
  `.apply(lambda v: int(v) if int(v) in {1..5} else 5)` clamp makes that
  structurally impossible.

Score: `correct − 0.25 × wrong − 1 × hallucinated`.

## 4. Manual checklist before submitting

The submitted zip contains **only** `setup.bash`, but `setup.bash` clones a
public GitHub repo, so the repo has to be ready first.

1. **Push this directory to a public GitHub repo.** Don't include
   `models/` (it's in `.gitignore`).
2. **Edit `setup.bash`:** replace `REPO_URL` and `REPO_BRANCH` with the
   actual values. Defaults:
   ```bash
   REPO_URL="https://github.com/REPLACE_USER/REPLACE_REPO.git"
   REPO_BRANCH="main"
   ```
3. **Verify the repo is public** before `2026-05-03 11:00 IST`.
4. **Build the zip.** It must contain only `setup.bash` (no folder
   wrapping it):
   ```bash
   zip project_2_<roll1>_<roll2>.zip setup.bash
   ```
   For a solo submission, follow the convention given by the TA.
5. **Smoke-test on a Linux box** (or WSL) before submitting:
   ```bash
   mkdir /tmp/grader_test && cd /tmp/grader_test
   unzip /path/to/project_2_<roll>.zip
   bash setup.bash
   conda activate gnr_project_env
   python inference.py --test_dir /absolute/path/to/sample_test_project_2
   cat submission.csv
   ```
   The CSV must have 3 columns and every `option` must be in `{1..5}`.

## 5. Local laptop smoke test (developer only — not used by grader)

`solver.py` exposes a CLI for laptop dev with the smaller 3B-AWQ model:

```bash
python download_model.py --variant 3b-awq
python solver.py \
    --model-path ./models/Qwen2.5-VL-3B-Instruct-AWQ \
    --test-csv  ../sample_test_project_2/test.csv \
    --image-dir ../sample_test_project_2/images \
    --out-csv   /tmp/smoke.csv \
    --num-samples 3 --dtype float16 --verbose
```

The 3B model is too small for accurate answers on math / code-tracing
questions — this is a *pipeline* check, not an *accuracy* check. The 32B-AWQ
that ships in the actual submission handles those cleanly on the L40s.

## 6. Design notes

- **Why a VLM, not OCR→LLM.** The pages contain LaTeX math and PyTorch code
  blocks. OCR on LaTeX / monospace is fragile — one botched subscript or
  mis-tokenized identifier corrupts downstream reasoning. A single VLM pass
  sees the rendered pixels end-to-end.
- **Why Qwen2.5-VL-32B-AWQ.** Strongest open-weight VLM at math + code +
  screenshot reasoning that fits in 48 GB VRAM (the L40s budget) with
  margin for a long generation context.
- **Self-consistency.** 3 samples at T=0.6, top_p=0.9, majority vote.
  Batched via `num_return_sequences` so the image prefill (the bulk of VLM
  latency) is amortized across samples.
- **Never hallucinate the output.** Parser extracts `ANSWER: <A|B|C|D>`
  with regex fallback to a trailing letter; any miss → 5. A final
  DataFrame-level clamp enforces `{1..5}`. The −1 penalty is 4× the
  per-wrong penalty — avoiding it is the single most important guard.
- **Always answer when a letter is parsed.** Random-guess EV over 4 options
  is `0.25·(+1) + 0.75·(−0.25) = +0.0625`. Skipping a parsed answer is
  net-negative; we only emit 5 on tied votes or total parse failure.
- **Compatibility shim.** `autoawq 0.2.x` imports `PytorchGELUTanh` from
  `transformers.activations`, renamed away in `transformers>=4.52`. Both
  `solver.py` and `inference.py` install a no-op shim before model load.

## 7. Citations

- Bai et al., *Qwen2.5-VL Technical Report*, 2025. Alibaba Qwen team.
  HF: `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`, `Qwen/Qwen2.5-VL-7B-Instruct`,
  `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`.
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in
  Language Models*, ICLR 2023.
- `transformers`, `accelerate`, `huggingface-hub`, `autoawq`, `qwen-vl-utils`
  — Apache-2.0 / MIT licenses. See `requirements.txt` for version floors.
