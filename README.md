# MuscleControl — MuJoCo 肌肉骨骼仿真平台

肌肉骨骼驱动的**行走仿真与控制**平台(MuJoCo CPU 刚体仿真 + 自定义 Hill 肌肉模型),目标是**偏瘫(hemiplegia)肌肉骨骼仿真验证**。技术路线为**纯直接优化**(无 RL / 不训练网络):

BVH/BIO 运动 → 参考姿态 q_ref → PD 期望力矩 → 逐帧优化 284 块肌肉激活 → Hill 肌肉 → 50 关节力矩 → MuJoCo 仿真

## 核心能力

- **直接肌肉控制**:284 块 Hill 肌肉逐帧反解激活,全身跟踪误差 ≈ 1.8°
- **MuJoCo 原生仿真**:CPU 刚性体(替代 Isaac Lab GPU),无需显卡
- **肌肉可视化**:284 肌肉线按激活灰→红着色;离屏视频 + 实时交互窗口
- **偏瘫实验**:患侧肌力减弱 80/60/40% 强度扫描
- **Tendon 验证**:waypoint vs MuJoCo 原生 spatial-tendon 几何对比(结论:保留 waypoint)

## 目录结构

| 路径 | 作用 |
|---|---|
| `debug/` | CLI 演示 / 渲染 / 数据工具(见下表) |
| `protomotions/utils/` | 核心:`direct_muscle_mujoco.py`(仿真)、`muscle_control.py`(Hill 肌肉)、`muscle_parser.py`(解析)、`direct_muscle.py`(共享) |
| `protomotions/simulator/isaaclab/` | 仅用于 Isaac 版对照 demo 的机器人定义 |
| `protomotions/data/assets/` | 模型资产:`muscle284.xml`、`mjcf/bio*.xml`、`mesh/` |
| `data/cmu_bio_npy/` | 输入运动(119 段 CMU 行走,BIO 骨架 .npy) |
| `mujoco_demo/tendon_prototype/` | waypoint vs tendon 对比验证(报告 `out/report.md`) |
| `mujoco_demo/viewer_demo/` | 实时交互 viewer(284 肌肉激活着色) |
| `output/` | 渲染产物(视频 / GIF / 轨迹 npz / 导出 CSV) |
| `archive/` | 归档:旧 RL 训练栈、历史调试、数据转换脚本 |

### `debug/` 工具清单

| 文件 | 作用 |
|---|---|
| `walk_muscle_demo_mujoco.py` | **主入口**:单段 / 批量评估、视频渲染、偏瘫强度扫描 |
| `render_muscle_mujoco.py` | 离线骨架渲染(npz → GIF/mp4,蓝=实际 红=参考) |
| `plot_activation.py` | 284 激活动态图(整段热图 PNG + 逐帧动画 GIF) |
| `export_traj.py` | **数据导出**:npz → CSV(q/q_ref/τ_des/τ_muscle/activation + summary) |
| `analyze_traj.py` | **数据分析**:CSV → 误差/对称性/激活热图/力矩匹配图 + 报告 |
| `walk_muscle_demo.py` | Isaac Lab 对照版(1.78° 复现凭证) |
| `test_muscle_parser.py` | 肌肉解析单测 |

## 环境

```bash
conda activate env_isaaclab    # Python 3.11, torch 2.7, mujoco 3.11
```

无 GPU 需求;渲染需 `imageio` / `imageio-ffmpeg`;交互窗口需 `glfw` / `PyOpenGL`。

## 快速开始

以下命令均在仓库根目录执行。

### 1. 单段评估

```bash
PYTHONPATH=. python debug/walk_muscle_demo_mujoco.py \
    --motion data/cmu_bio_npy/009/09_12.npy
```

### 2. 批量评估(119 段)

```bash
PYTHONPATH=. python debug/walk_muscle_demo_mujoco.py \
    --motion "data/cmu_bio_npy/*/*.npy" --batch
```

### 3. 渲染 10 秒肌肉行走视频(60fps, 网格地面, 全身)

```bash
PYTHONPATH=. python debug/walk_muscle_demo_mujoco.py \
    --motion data/cmu_bio_npy/009/09_12.npy --max-frames 300 \
    --render-video output/mj_muscle_walk.mp4 \
    --render-width 1280 --render-height 960
```

### 4. 284 肌肉激活图(热图 + 动画 GIF)

```bash
PYTHONPATH=. python debug/walk_muscle_demo_mujoco.py \
    --motion data/cmu_bio_npy/009/09_12.npy --max-frames 300 --save-npz output/mj_traj.npz
PYTHONPATH=. python debug/plot_activation.py --npz output/mj_traj.npz \
    --png output/activation_heatmap.png --gif output/activation_anim.gif
```

### 5. 实时交互窗口(284 肌肉激活着色)

```bash
PYTHONPATH=. python mujoco_demo/viewer_demo/view_demo_muscles.py --loop
```

### 6. 数据导出与分析

```bash
# 导出轨迹为 CSV (q/q_ref/τ_des/τ_muscle/activation + summary.json)
PYTHONPATH=. python debug/walk_muscle_demo_mujoco.py \
    --motion data/cmu_bio_npy/009/09_12.npy --max-frames 300 --save-npz output/mj_traj.npz
PYTHONPATH=. python debug/export_traj.py --npz output/mj_traj.npz --out output/export

# 自动分析 (误差/对称性/激活热图/力矩匹配 + report.txt)
PYTHONPATH=. python debug/analyze_traj.py --dir output/export
```

## 偏瘫实验(患侧肌力强度扫描)

```bash
PYTHONPATH=. python debug/walk_muscle_demo_mujoco.py \
    --motion data/cmu_bio_npy/009/09_12.npy --affected-side L --strength-sweep
```

> 实验结论:逐帧全知优化下,患侧肌力降到 40% 跟踪仍几乎不变(激活自动补偿)。
> 偏瘫步态异常主要由痉挛 / 神经驱动受限等机制驱动,平台后续在此方向扩展。

## Tendon 验证(并行, 不改生产)

```bash
PYTHONPATH=. python -u mujoco_demo/tendon_prototype/compare.py    # 对比报告
PYTHONPATH=. python -u mujoco_demo/tendon_prototype/visualize.py  # 双路径可视化
```

结论:**KEEP_WAYPOINT_BACKEND**(步态 ROM 内差异 <7mm,无系统性符号反向)。

## 归档(archive/)

RL 训练栈、历史调试脚本、数据转换管线已移至 `archive/`(git 历史可恢复),详见 `archive/README.md`。

## 关键技术点

- 力矩臂口径 τ=f·r:生产 `JtA` 已含负号;MuJoCo `ten_velocity` 取负对齐。
- MuJoCo 需 `dof_armature=0.03` 防自由关节发散;激活优化用 `lbfgs + 50` 迭代。
- 3-DOF 关节用 `quat_to_exp_map` 转 DOF(避免 euler 的 2π 环绕)。

## 历史对照

- `debug/walk_muscle_demo.py`:Isaac Lab 版(1.78°,需 GPU / Isaac Sim),仅作复现凭证。

