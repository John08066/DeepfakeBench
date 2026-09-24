#!/bin/bash
set -eu
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$project_root"
python_bin=${PYTHON_BIN:-python}

# ============================================================
# DeepfakeBench Training Launcher
#
# 使用方法：
#   1. 只保留一个训练方案处于非注释状态
#   2. 其他方案全部保持 # 注释
#   3. 在 DeepfakeBench 项目根目录执行：
#
#      bash train.sh
#
# 长时间训练建议在 tmux 中运行。
# ============================================================


# ============================================================
# 方案 1：单 GPU 训练
# 默认启用
# 适合调试、smoke test、小规模实验
# ============================================================

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
"$python_bin" training/train.py \
  --detector_path ./training/config/detector/prd_probe_ablation.yaml "$@"


# ============================================================
# 方案 2：4 GPU DDP —— 推荐的标准多卡训练方式

# ============================================================

# torchrun \
#   --nproc_per_node=4 \
#   training/train.py \
#   --detector_path ./training/config/detector/sbi.yaml \
#   --no-save_ckpt \
#   --no-save_feat \
#   --ddp


# ============================================================
# 方案 3：指定 GPU 1、2、3，使用 3 GPU DDP
#
# 注意：
# CUDA_VISIBLE_DEVICES=1,2,3 后，
# 程序内部看到的是逻辑 GPU 0,1,2
# ============================================================

# CUDA_VISIBLE_DEVICES=1,2,3 \
# torchrun \
#   --nproc_per_node=3 \
#   training/train.py \
#   --detector_path ./training/config/detector/sbi.yaml \
#   --no-save_ckpt \
#   --no-save_feat \
#   --ddp


# ============================================================
# 方案 4：4 GPU DDP，并保存 checkpoint / feature
#
# 和方案 2 的区别：
# 不加 --no-save_ckpt
# 不加 --no-save_feat
# ============================================================

# torchrun \
#   --nproc_per_node=4 \
#   training/train.py \
#   --detector_path ./training/config/detector/sbi.yaml \
#   --ddp


# ============================================================
# 方案 5：旧版 PyTorch distributed.launch
#
# 仅用于兼容旧环境。
# 新版 PyTorch 优先使用 torchrun。
# ============================================================

# python -m torch.distributed.launch \
#   --nproc_per_node=4 \
#   training/train.py \
#   --detector_path ./training/config/detector/sbi.yaml \
#   --no-save_ckpt \
#   --no-save_feat \
#   --ddp


# ============================================================
# 方案 6：后台 nohup 方式
#
# 不推荐作为我们的正式训练方式。
# 长训练统一优先使用 tmux。
# 这里保留仅用于理解/兼容旧脚本。
# ============================================================

# nohup python3 -m torch.distributed.launch \
#   --nproc_per_node=4 \
#   training/train.py \
#   --detector_path ./training/config/detector/sbi.yaml \
#   --no-save_ckpt \
#   --no-save_feat \
#   --ddp \
#   > my_output.log 2>&1 &#!/bin/bash

# ============================================================
