#!/bin/bash
# =============================================================================
# Phase 1: PPO PD 行走专家训练 (教师)
# 目标: 训练一个用 PD 控制跟踪 walking_bio.pt 的行走专家策略,
#       作为 Phase 2 肌肉学生的教师 (expert)。
#
# 云桌面环境: 1x RTX 5880 16GB, Isaac Sim 5.0, Isaac Lab 2.2.0, Ubuntu 22.04
#
# 用法:
#   bash scripts/train_phase1.sh
#   WANDB_ID=xxx bash scripts/train_phase1.sh        # 启用 wandb 断点续训
#   NUM_ENVS=256 bash scripts/train_phase1.sh        # 显存不足时调小
# =============================================================================
set -e

# ---- 可调参数 ----
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"          # conda 环境名
NUM_ENVS="${NUM_ENVS:-512}"                       # 并行环境数 (16GB 显存推荐 512)
BATCH_SIZE="${BATCH_SIZE:-2048}"                  # PPO batch size
EXP_NAME="${EXP_NAME:-walking_pd_expert}"         # 实验名 (产物在 results/<EXP_NAME>/)
WANDB_ID="${WANDB_ID:-null}"                      # wandb run id (续训用)

cd "$(dirname "$0")/.."                           # 进入仓库根目录

# ---- 数据完整性检查: walking_bio.pt 必须是真实数据 (zip 格式, 不是 Git LFS 指针) ----
if [ "$(head -c 2 data/walking_bio.pt 2>/dev/null | od -An -tx1 | tr -d ' \n')" != "504b" ]; then
  echo "data/walking_bio.pt 缺失或损坏 (可能仍是 LFS 指针), 执行 git lfs pull 拉取真实数据..."
  git lfs pull
  if [ "$(head -c 2 data/walking_bio.pt | od -An -tx1 | tr -d ' \n')" != "504b" ]; then
    echo "错误: data/walking_bio.pt 拉取失败, 请手动执行: git lfs pull"
    exit 1
  fi
fi
echo "data/walking_bio.pt 校验通过 ($(du -h data/walking_bio.pt | cut -f1))"

# ---- 激活 conda 环境 ----
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || { echo "未找到 conda, 请修改脚本里的 CONDA 激活路径"; exit 1; }
set +e   # conda activate 内部命令在 set -e 下会误触发退出
conda activate "$CONDA_ENV"
set -e
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"   # 必须在 activate 之后设置 (conda 会重置 PYTHONPATH)

echo "=== Phase 1: PPO PD 行走专家 ==="
echo "  envs=$NUM_ENVS  batch=$BATCH_SIZE  exp=$EXP_NAME"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 减少显存碎片 (8GB 小卡)
CUDA_VISIBLE_DEVICES=0 python protomotions/train_agent.py \
    +exp=full_body_tracker/transformer_flat_terrain \
    +robot=bio_act \
    +simulator=isaaclab \
    motion_file=./data/walking_bio.pt \
    +experiment_name="$EXP_NAME" \
    num_envs="$NUM_ENVS" \
    agent.config.batch_size="$BATCH_SIZE" \
    agent.config.num_mini_epochs=2 \
    agent.config.eval_metrics_every=2000 \
    +agent.config.train_teacher=true \
    ngpu=1

# 如需 wandb 记录, 在训练命令后追加:
#   +opt=wandb wandb.wandb_id=${WANDB_ID} wandb.wandb_resume=allow
