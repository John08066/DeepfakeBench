#!/usr/bin/env bash
# Shared sequential launcher for the already-approved PRD experiment queue.
set -eu
cd /home/zhaoting.ding/disk/zhaoting.ding/DeepfakeBench
test "$#" -eq 2 || { echo 'Usage: launch_prd_training.sh CONFIG TASK'; exit 2; }
config=$1
task=$2
test -f "$config"
[[ "$task" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
mkdir -p .state
exec 9>.state/prd_training.lock
flock -n 9 || { echo 'Another PRD launcher holds the training lock.'; exit 1; }
/usr/bin/python3 -c 'import sys; sys.path.insert(0, "scripts"); from watch_prd_early_stop import project_training; p = project_training(); print("Existing training PIDs:", p); sys.exit(bool(p))'
mkdir -p "logs/$task"
set -o noclobber
exec >"logs/$task/console.log" 2>&1
echo "Training launch: $(date -Is); config=$config; task=$task; GPU=0"
set +e
DEEPFAKE_DATA_ROOT=/home/zhaoting.ding/local_datasets \
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /home/zhaoting.ding/miniconda3/envs/PRD/bin/python -u training/train.py \
    --detector_path "$config" --task_target "$task"
code=$?
echo "TRAINING_EXIT_CODE=$code at $(date -Is)"
exit "$code"
