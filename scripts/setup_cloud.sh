#!/bin/bash
# =============================================================================
# 云桌面环境安装脚本
# 目标环境: 1x RTX 5880 16GB, Isaac Sim 5.0, Isaac Lab 2.2.0, Ubuntu 22.04
#
# 完成: 检查 walking_bio.pt -> 安装 ProtoMotions 依赖 -> 验证
#
# 用法 (在仓库根目录):
#   ISAACLAB_PATH=/path/to/IsaacLab bash scripts/setup_cloud.sh
#
# 若尚未 clone 仓库, 先执行:
#   git clone https://github.com/rongqi05/muscle-pt.git
#   cd muscle-pt
#   # 然后手动下载数据 (从 GitHub Release):
#   #   curl -L -H "Authorization: Bearer <PAT>" -o data/walking_bio.pt \
#   #     https://github.com/rongqi05/muscle-pt/releases/download/data-v2/walking_bio.pt
# =============================================================================
set -e

ISAACLAB_PATH="${ISAACLAB_PATH:?请设置 ISAACLAB_PATH 环境变量 (Isaac Lab 安装目录, 含 isaaclab.sh)}"

cd "$(dirname "$0")/.."   # 进入仓库根

echo "=== [1/3] 检查训练数据 (walking_bio.pt) ==="
if [ ! -f data/walking_bio.pt ] || [ "$(head -c 2 data/walking_bio.pt 2>/dev/null | od -An -tx1 | tr -d ' \n')" != "504b" ]; then
  echo "错误: 缺少有效的 data/walking_bio.pt (约 161MB)。"
  echo "请从 GitHub Release 下载 (需 PAT):"
  echo "  curl -L -H \"Authorization: Bearer <PAT>\" -o data/walking_bio.pt \\"
  echo "    https://github.com/rongqi05/muscle-pt/releases/download/data-v2/walking_bio.pt"
  exit 1
fi
echo "数据检查通过: $(du -h data/walking_bio.pt | cut -f1)"

echo "=== [2/3] 安装依赖 (用 Isaac Lab 的 python) ==="
# 注: protomotions 包没有 setup.py/pyproject.toml, 不执行 `pip install -e .`;
# 它靠"从仓库根目录运行"导入 (python 的 sys.path 含 cwd)。
"$ISAACLAB_PATH/isaaclab.sh" -p -m pip install -r requirements_isaaclab.txt
"$ISAACLAB_PATH/isaaclab.sh" -p -m pip install -e isaac_utils
"$ISAACLAB_PATH/isaaclab.sh" -p -m pip install -e poselib

echo "=== [3/3] 验证 ==="
echo "-- walking_bio.pt --"
ls -la data/walking_bio.pt
if head -c 100 data/walking_bio.pt | grep -q "git-lfs"; then
  echo "错误: walking_bio.pt 不是有效数据, 请重新下载"
  exit 1
fi
echo "-- import 检查 --"
PYTHONPATH=. "$ISAACLAB_PATH/isaaclab.sh" -p -c "import protomotions, poselib, isaaclab; print('protomotions/poselib/isaaclab import OK')"

echo ""
echo "=== 安装完成 ==="
echo "下一步:"
echo "  bash scripts/train_phase1.sh   # Phase 1 PD 行走专家"
echo "  bash scripts/train_phase2.sh   # Phase 2 肌肉学生 (Phase 1 完成后)"
