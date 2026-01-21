#!/usr/bin/env bash

set -euo pipefail

# === Resolve absolute paths relative to script ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

image="${REPO_ROOT}/containers/tensorflow_2.14.0-gpu.sif" # <=== Adjust as needed
if [[ ! -f "$image" ]]; then
  echo "ERROR: image not found: $image" >&2
  echo "Please download the SIF file and try again." >&2
  exit 1
fi

# === Add code repos to PYTHONPATH (prefer repo/src if it exists) ===
git_dir="${REPO_ROOT}/ext" # <=== Adjust as needed
git_rep=(
  "${git_dir}/pystrum"
  "${git_dir}/neurite"
  "${git_dir}/voxelmorph"
)

d="${PYTHONPATH:-}"
for r in "${git_rep[@]}"; do
  if [[ -d "$r/src" ]]; then
    part="$r/src"
  elif [[ -d "$r" ]]; then
    part="$r"
  else
    continue
  fi
  d="${d:+$d:}$part"
done
export APPTAINERENV_PYTHONPATH="$d"

# === Base binds array ===
BIND_OPTS=( -B /autofs ) # <=== Adjust as needed

# === Run the container ===
apptainer exec --nv "${BIND_OPTS[@]}" -e "$image" "$@"
