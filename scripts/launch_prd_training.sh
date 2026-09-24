#!/usr/bin/env bash
# Launch one explicitly selected experiment in this checkout.
set -eu
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
python_bin=${PYTHON_BIN:-python}
test "$#" -eq 2 || { echo 'Usage: launch_prd_training.sh CONFIG TASK'; exit 2; }
config=$1
task=$2
test -f "$config"
[[ "$task" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
mkdir -p .state
exec 9>.state/prd_training.lock
flock -n 9 || { echo 'Another PRD launcher holds the training lock.'; exit 1; }
"$python_bin" -c 'import sys; sys.path.insert(0, "scripts"); from watch_prd_early_stop import project_training; p = project_training(); print("Existing training PIDs:", p); sys.exit(bool(p))'
console_dir=$("$python_bin" -c 'import sys; from training.path_config import resolve_output_path; print(resolve_output_path(sys.argv[1]))' "logs/$task")
mkdir -p "$console_dir"
set -o noclobber
exec 8>"$console_dir/console.log"  # Refuse to overwrite an existing run log.
set +e
{
    echo "Training launch: $(date -Is); config=$config; task=$task; GPU=${CUDA_VISIBLE_DEVICES:-0}"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
        "$python_bin" -u training/train.py \
        --detector_path "$config" --task_target "$task"
    code=$?
    echo "TRAINING_EXIT_CODE=$code at $(date -Is)"
    exit "$code"
} 2>&1 | tee /dev/fd/8  # Keep Python non-interactive while mirroring output to tmux.
status=("${PIPESTATUS[@]}")
test "${status[0]}" -eq 0 || exit "${status[0]}"  # Preserve the training exit code.
exit "${status[1]}"  # Surface a logging failure when training itself succeeded.
