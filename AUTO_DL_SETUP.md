# AutoDL 从零部署 Isaac Sim + Isaac Lab 训练环境

目标机器: AutoDL 084 机(RTX 4090D 24GB 直通, 驱动 580.76.05, CUDA 13.0, 预装 PyTorch 2.7.0)

> 本机(能跑通的参考环境)安装方式: Isaac Sim 5.1.0 为 pip 安装, Isaac Lab 2.3.2 为源码 clone + editable 安装。以下步骤完全复现该方式。

---

## 0. 前置检查

```bash
# 确认 GPU 直通 (4090D 应显示完整名称, 无 vGPU 字样)
nvidia-smi

# 确认预装 torch 的 CUDA 版本 (Isaac Sim 5.1 需要 cu128)
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# 磁盘空间 (Isaac Sim 约 30-40GB, 建议数据盘 ≥ 60GB)
df -h
```

若 torch 的 `cuda` 不是 `12.8`(cu128), 先重装:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

---

## 1. 创建 conda 环境

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
```

## 2. 安装 Isaac Sim 5.1.0(pip, NVIDIA 官方源)

```bash
pip install isaacsim==5.1.0.0 --extra-index-url https://pypi.nvidia.com
```

> 会拉取 `isaacsim-kernel`、`isaacsim-app` 等全部子包, 约 30-40GB, 耐心等待。
> 验证: `python -c "import isaacsim; print('isaacsim OK')"`

## 3. 克隆 IsaacLab 源码 v2.3.2

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2
```

## 4. 安装 IsaacLab 扩展(官方脚本)

```bash
cd IsaacLab
./isaaclab.sh -i          # 装系统依赖 + source/ 下所有扩展 + rl 框架
```

> `-i` 会交互式询问, 全部回车默认即可。完成后验证:
> ```bash
> ./isaaclab.sh -p -c "import isaaclab; print('isaaclab OK', isaaclab.__version__)"
> ```

---

## 5. 拉取训练仓库 + 数据

```bash
cd ~
git clone https://github.com/rongqi05/muscle-pt.git
cd muscle-pt

# 拉取训练数据 (161MB, 公开 LFS 免认证)
sudo apt install git-lfs   # 或 conda install -c conda-forge git-lfs
git lfs install
git lfs pull

# 验证: 应约 161MB
ls -l data/walking_bio.pt
```

## 6. 安装 ProtoMotions 依赖

```bash
cd ~/muscle-pt
ISAACLAB_PATH=~/IsaacLab bash scripts/setup_cloud.sh
```

> 完成后应打印 `protomotions/poselib/isaaclab import OK`。

## 7. 验证 Isaac Sim 物理引擎(GPU 直通应通过)

```bash
cd ~/IsaacLab
python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import isaaclab.sim as sim_utils
sim_cfg = sim_utils.SimulationCfg(device='cuda:0', dt=1/60)
sim = sim_utils.SimulationContext(sim_cfg)
sim.reset()
print('>>> PhysX OK')
app.close()
"
```

看到 `>>> PhysX OK` 即可训练。

## 8. 训练

```bash
cd ~/muscle-pt
bash scripts/train_phase1.sh      # Phase 1 PD 专家 (num_envs=1024, batch=4096)
# Phase 1 完成后:
bash scripts/train_phase2.sh      # Phase 2 肌肉学生
```

---

## 常见坑

| 症状 | 原因 | 解决 |
|---|---|---|
| `CUDA driver error: operation not supported` | vGPU 虚拟化(不应出现, 4090D 是直通) | 确认是直通实例 |
| torch cuda 版本不是 cu128 | AutoDL 预装 cu121/cu124 | 重装 `torch==2.7.0` 到 cu128 |
| `protobuf` 版本冲突 | tensorboard 2.21 vs protobuf 5.x | `pip install tensorboard==2.18.0` |
| 磁盘不足 | isaacsim 30-40GB + conda | 扩容数据盘或用 `/root/autodl-tmp` |
| `ModuleNotFoundError: protomotions` | PYTHONPATH 被 conda 重置 | 脚本已修复, 或手动 `export PYTHONPATH=$(pwd)` |

---

## 磁盘空间建议

- Isaac Sim 5.1.0(pip): ~30-40GB
- IsaacLab 源码 + editable 安装: ~2GB
- conda 环境(torch 等): ~5GB
- 训练数据 + checkpoint: ~1GB

**合计约 45-50GB**, 你的 50GB 数据盘很紧。建议:
1. 扩容到 76GB(你提到可扩容 26GB), 或
2. 把 Isaac Sim 的 pip 缓存/大文件放 `/root/autodl-tmp`
