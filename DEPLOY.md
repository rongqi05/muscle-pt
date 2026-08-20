# 云端训练部署指南

在云服务器上复现肌肉驱动人形行走训练(Phase 1 PD 专家 → Phase 2 肌肉学生)。

---

## 0. 前置要求(选云服务器前必读)

### 0.1 GPU 类型 —— 最关键的一步

Isaac Sim 的 PhysX 物理引擎需要底层 CUDA-Vulkan 互操作,**只有 GPU 直通(透传)实例才能跑**。

| GPU 类型 | 能否训练 | 说明 |
|---|---|---|
| ✅ **GPU 直通 (passthrough)** | 能 | 整卡独享, Isaac Sim 完全支持 |
| ❌ **vGPU / 共享 GPU** | 不能 | 报 `CUDA driver error: operation not supported` + `NVML_ERROR_NOT_SUPPORTED` |

> 教训: 之前一台 RTX 5880 Ada (vGPU 虚拟化) 上 PhysX 初始化失败, 排查很久最终确认是 vGPU 硬限制。

### 0.2 推荐配置

- **GPU**: RTX 4090 / 4090D / A6000 / A100(Ada/Ampere 架构)
- ⚠️ **避开 Blackwell (RTX 50 系)**: Isaac Sim 5.x 渲染/驱动与 sm_120 架构不兼容
- **显存**: ≥ 16GB(训练 512 环境 + 284 肌肉 Jacobian)
- **系统**: Ubuntu 22.04
- **驱动**: NVIDIA 535+
- **磁盘**: ≥ 100GB(Isaac Sim 本体约 30-50GB)

### 0.3 选好机器后, 先验证 Isaac Sim 能否初始化

```bash
# STEP 1: 只测 SimulationApp 能否启动 (不碰 PhysX)
python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
print('STEP1_OK: SimulationApp 启动成功')
app.close()
"

# STEP 2: 测 PhysX 物理上下文能否创建 (vGPU 会在这步崩)
python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import isaaclab.sim as sim_utils
sim_cfg = sim_utils.SimulationCfg(device='cuda:0', dt=1/60)
sim = sim_utils.SimulationContext(sim_cfg)
sim.reset()
print('STEP2_OK: PhysX SimulationContext 创建成功')
app.close()
"
```

- 两条都输出 `OK` → 可以继续部署
- 只有 STEP1_OK, STEP2 报 `CUDA driver error: operation not supported` → **换 GPU 直通实例**

---

## 1. 拉取代码

仓库已公开, 免认证:

```bash
git clone https://github.com/rongqi05/muscle-pt.git
cd muscle-pt
```

## 2. 拉取训练数据 (Git LFS)

```bash
# 安装 git-lfs (如未安装)
sudo apt install git-lfs        # 或  conda install -c conda-forge git-lfs
git lfs install

# 拉取 walking_bio.pt (161MB)
git lfs pull

# 验证: 应约 161MB, 开头为 zip 魔数 PK
ls -l data/walking_bio.pt
head -c 2 data/walking_bio.pt | od -c
```

## 3. 安装依赖

需要一个已装好 Isaac Lab 的环境(conda 环境名默认 `env_isaaclab`), 并知道 `isaaclab.sh` 路径:

```bash
# 找到 isaaclab.sh (常见在 ~/IsaacLab 或 /opt/IsaacLab)
find ~ /opt -maxdepth 3 -name "isaaclab.sh" 2>/dev/null

# 用找到的路径替换 <路径>, 安装 protomotions 依赖
ISAACLAB_PATH=<路径> bash scripts/setup_cloud.sh
```

> 完成后应打印 `protomotions/poselib/isaaclab import OK`。

### 3.1 已知依赖坑: tensorboard 降级

若训练时报 `protobuf` 版本冲突(gencode 6.x / runtime 5.x), 降级 tensorboard:

```bash
pip install "tensorboard==2.18.0"
```

## 4. 训练 Phase 1(PD 行走专家)

```bash
bash scripts/train_phase1.sh
```

默认参数(适合 16GB 直通 GPU):

| 参数 | 默认值 | 说明 |
|---|---|---|
| `NUM_ENVS` | 512 | 并行环境数, 显存不足可降 256 |
| `BATCH_SIZE` | 2048 | PPO batch size |
| `EXP_NAME` | walking_pd_expert | 产物目录 |
| `PHYSX_GPU` | true | GPU PhysX, 直通 GPU 保持 true |

```bash
# 显存不足时
NUM_ENVS=256 bash scripts/train_phase1.sh
```

- 产物: `results/walking_pd_expert/last.ckpt`(定期自动保存, 每 10 epoch)
- 用 tensorboard 看曲线: `tensorboard --logdir results/walking_pd_expert`
- 训练无步数上限, 观察 `total_rew` 收敛(0.9+)后 `Ctrl+C` 停止

## 5. 训练 Phase 2(284 肌肉学生)

Phase 1 完成后:

```bash
bash scripts/train_phase2.sh
```

- 自动读 `results/walking_pd_expert/` 当教师
- 产物: `results/walking_muscle_student/`

## 6. 回放评估(可选)

```bash
# headless 采集策略轨迹 + matplotlib 渲染 GIF (不依赖 Isaac Sim GUI)
python debug/replay_policy.py \
  --checkpoint results/walking_pd_expert/last.ckpt \
  --steps 200 --num-envs 4 \
  --gif output/replay.gif --frames output/replay_frames

# 或用已保存轨迹单独重渲染
python debug/render_replay.py --npz output/replay_traj.npz --gif output/replay.gif
```

> 注: 回放脚本专为无 GUI 环境设计 (本机 RTX 5060 渲染不可用时的替代方案), 云服务器若 GUI 可用可直接用 `eval_agent.py` 开窗口看。

---

## 常见问题排查

| 症状 | 原因 | 解决 |
|---|---|---|
| `CUDA driver error: operation not supported` | vGPU 虚拟化 | 换 GPU 直通实例 |
| `CUDA out of memory` | 显存不足 | 降 `NUM_ENVS` |
| 评估阶段(每 2000 epoch)崩 | 8GB 小卡评估 buffer 溢出 | 已默认跳过评估 (`EVAL_EVERY`) |
| `ModuleNotFoundError: protomotions` | PYTHONPATH 被 conda 重置 | 脚本已修复(activate 后设置) |
| `protobuf` 版本冲突 | tensorboard 2.21 与 protobuf 5.x | `pip install tensorboard==2.18.0` |

---

## 从本地 checkpoint 续训(可选)

本地已训到 epoch 9870 的 checkpoint, 想云端续训:

```bash
# 本地: 上传 checkpoint 到仓库 (需 LFS 或手动传)
# 云端: 放到 results/walking_pd_expert/ 后直接跑 train_phase1.sh (会自动续训)
```
