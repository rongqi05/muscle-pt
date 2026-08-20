#!/bin/bash
# =============================================================================
# Phase 2: 284 维肌肉学生训练
# 目标: 训练一个输出 284 个肌肉激活的策略, 逼近 Phase 1 教师的 PD 扭矩。
#
# 前置条件: 先跑完 Phase 1 (scripts/train_phase1.sh),
#           产物在 results/walking_pd_expert/ (含 score_based.ckpt / last.ckpt)
#
# AutoDL 环境: RTX 4090D 24GB (GPU 直通), Ubuntu 22.04, 驱动 580
#
# 用法:
#   bash scripts/train_phase2.sh
#   EXPERT=results/<别的实验名> bash scripts/train_phase2.sh
# =============================================================================
set -e

# ---- 可调参数 ----
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
NUM_ENVS="${NUM_ENVS:-1024}"                      # 并行环境数 (24GB 显存推荐 1024)
BATCH_SIZE="${BATCH_SIZE:-2048}"                  # PPO batch size
EXP_NAME="${EXP_NAME:-walking_muscle_student}"
EXPERT="${EXPERT:-results/walking_pd_expert}"      # Phase 1 教师 checkpoint 目录
PHYSX_GPU="${PHYSX_GPU:-true}"                    # GPU PhysX; 4090D 直通保持 true

cd "$(dirname "$0")/.."

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

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || { echo "未找到 conda, 请修改脚本里的 CONDA 激活路径"; exit 1; }
set +e   # conda activate 内部命令在 set -e 下会误触发退出
conda activate "$CONDA_ENV"
set -e
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"   # 必须在 activate 之后设置 (conda 会重置 PYTHONPATH)

if [ ! -d "$EXPERT" ]; then
  echo "错误: 未找到教师 checkpoint 目录 $EXPERT"
  echo "请先运行 scripts/train_phase1.sh"
  exit 1
fi

echo "=== Phase 2: 284 维肌肉学生 ==="
echo "  envs=$NUM_ENVS  batch=$BATCH_SIZE  exp=$EXP_NAME  expert=$EXPERT  physx_gpu=$PHYSX_GPU"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 减少显存碎片 (8GB 小卡)
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
    simulator.config.sim.physx.use_gpu="$PHYSX_GPU" \
    ngpu=1
