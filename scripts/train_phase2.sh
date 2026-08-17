#!/bin/bash
# =============================================================================
# Phase 2: 284 维肌肉学生训练
# 目标: 训练一个输出 284 个肌肉激活的策略, 逼近 Phase 1 教师的 PD 扭矩。
#
# 前置条件: 先跑完 Phase 1 (scripts/train_phase1.sh),
#           产物在 results/walking_pd_expert/ (含 score_based.ckpt / last.ckpt)
#
# 云桌面环境: 1x RTX 5880 16GB, Isaac Sim 5.0, Isaac Lab 2.2.0, Ubuntu 22.04
#
# 用法:
#   bash scripts/train_phase2.sh
#   EXPERT=results/<别的实验名> bash scripts/train_phase2.sh
# =============================================================================
set -e

# ---- 可调参数 ----
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
NUM_ENVS="${NUM_ENVS:-512}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
EXP_NAME="${EXP_NAME:-walking_muscle_student}"
EXPERT="${EXPERT:-results/walking_pd_expert}"      # Phase 1 教师 checkpoint 目录

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"   # 仓库根加入 PYTHONPATH, 供 import protomotions

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || { echo "未找到 conda, 请修改脚本里的 CONDA 激活路径"; exit 1; }
conda activate "$CONDA_ENV"

if [ ! -d "$EXPERT" ]; then
  echo "错误: 未找到教师 checkpoint 目录 $EXPERT"
  echo "请先运行 scripts/train_phase1.sh"
  exit 1
fi

echo "=== Phase 2: 284 维肌肉学生 ==="
echo "  envs=$NUM_ENVS  batch=$BATCH_SIZE  exp=$EXP_NAME  expert=$EXPERT"

CUDA_VISIBLE_DEVICES=0 python protomotions/train_agent.py \
    +exp=mus/no_vae_no_text_flat_terrain \
    +robot=bio_act_stu \
    +simulator=isaaclab \
    motion_file=./data/walking_bio.pt \
    +experiment_name="$EXP_NAME" \
    agent.config.expert_model_path="$EXPERT" \
    num_envs="$NUM_ENVS" \
    agent.config.batch_size="$BATCH_SIZE" \
    agent.config.num_mini_epochs=2 \
    ngpu=1
