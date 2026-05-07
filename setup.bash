#!/usr/bin/env bash
# setup.bash — environment + code + weight provisioning.
#
# Grading workflow (per project guidelines):
#     cd ./your_directory
#     bash setup.bash                  <-- this script (internet ON)
#     conda activate gnr_project_env
#     python inference.py --test_dir <absolute_path_to_test_dir>   (internet OFF)
#
# This script is the SOLE file inside the submitted zip. It clones the project's
# public GitHub repo into the current directory, then creates the conda env
# `gnr_project_env` (python 3.11), installs deps, and downloads model weights.

set -euo pipefail

# --------- USER CONFIG ----------------------------------------------------
# Public GitHub repo containing inference.py, solver.py, requirements.txt,
# download_model.py. MUST be public and pushed BEFORE the grading window opens.
REPO_URL="https://github.com/piyushll/gnr638-project.git"
REPO_BRANCH="main"
ENV_NAME="gnr_project_env"
PY_VERSION="3.11"
MODEL_VARIANT_PRIMARY="7b"                # Qwen2.5-VL-7B-Instruct (non-AWQ, ~17 GB) — avoids the autoawq Marlin kernel SM89 mismatch on L40s
# --------------------------------------------------------------------------

echo "[setup] $(date -Is) starting"
echo "[setup] cwd: $(pwd)"

# 0. Hard prerequisites: conda and git must be on PATH. Fail loudly if not.
command -v conda >/dev/null 2>&1 || { echo "[setup] ERROR: conda not on PATH" >&2; exit 1; }
command -v git   >/dev/null 2>&1 || { echo "[setup] ERROR: git not on PATH"   >&2; exit 1; }

# 1. Clone the repo into a temp dir, then copy its contents into cwd.
#    We can't `git clone` directly into cwd because setup.bash already lives here.
TMP_REPO="$(mktemp -d -t gnr_repo_XXXXXX)"
trap 'rm -rf "$TMP_REPO"' EXIT
echo "[setup] cloning $REPO_URL ($REPO_BRANCH) -> $TMP_REPO"
git clone --depth=1 --branch "$REPO_BRANCH" "$REPO_URL" "$TMP_REPO"

echo "[setup] copying repo files into cwd (skipping .git/ and any cloned setup.bash)"
shopt -s dotglob nullglob
for entry in "$TMP_REPO"/*; do
    base="$(basename "$entry")"
    case "$base" in
        .git|setup.bash) continue ;;
    esac
    # -rf so a re-run of setup.bash doesn't fail on pre-existing files.
    cp -rf "$entry" "./$base"
done
shopt -u dotglob

# 2. Create conda env (python 3.11) and install deps with `conda run`.
#    Using `conda run` avoids needing to source conda.sh for an inline activate.
echo "[setup] creating conda env: $ENV_NAME (python=$PY_VERSION)"
# -y is idempotent w.r.t. acceptance prompts but not w.r.t. existing envs.
# If env already exists, conda create errors; tear it down first to be safe.
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
conda create -n "$ENV_NAME" "python=$PY_VERSION" -y

# `conda run --live-stream` is conda 22.11+. If unavailable, fall back to
# plain `conda run` (output is buffered but still appears).
CONDA_RUN=(conda run -n "$ENV_NAME")
if conda run --help 2>/dev/null | grep -q -- '--live-stream'; then
    CONDA_RUN=(conda run --live-stream -n "$ENV_NAME")
fi

echo "[setup] installing pip dependencies into $ENV_NAME"
"${CONDA_RUN[@]}" python -m pip install --upgrade pip
"${CONDA_RUN[@]}" python -m pip install --no-cache-dir -r requirements.txt

# 3. Download model weights (~17 GB). One model only — the 7B non-AWQ load
#    path is the verified-working configuration on the L40s grader.
echo "[setup] downloading PRIMARY model: $MODEL_VARIANT_PRIMARY"
"${CONDA_RUN[@]}" python download_model.py \
    --variant "$MODEL_VARIANT_PRIMARY" --out-dir ./models

# 4. Sanity check: inference.py exists and the model directory is populated.
test -f ./inference.py || { echo "[setup] ERROR: inference.py missing after clone" >&2; exit 1; }
test -d ./models || { echo "[setup] ERROR: ./models/ missing after download" >&2; exit 1; }
echo "[setup] $(date -Is) done"
