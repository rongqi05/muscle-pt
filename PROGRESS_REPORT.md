# 阶段性成果报告 —— MuJoCo 肌肉骨骼仿真平台

> 更新: 2026-08-29
> 目标: 在 MuJoCo 中构建**偏瘫(hemiplegia)人体肌肉骨骼仿真验证平台**,技术基础为 284 块 Hill 肌肉驱动的行走仿真。

---

## 1. 技术路线(最终方案)

**纯直接优化**(不训练网络、无 RL),逐帧反解肌肉激活;仿真后端为 **MuJoCo CPU 刚体引擎**(替代 Isaac Lab GPU):

```
BVH/BIO 运动 → q_ref → PD 期望力矩 τ_des → 优化 284 激活 a
    → Hill 肌肉 (a·JtA + b) → 50 关节力矩 → MuJoCo 仿真
```

每个物理子步(240Hz):
1. PD 控制器:`τ_des = kp·(q_ref − q) − kd·qd`;
2. LBFGS 解 284 激活使 `a·JtA + b ≈ τ_des`;
3. Hill 肌肉模型输出 50 关节力矩 → `qfrc_applied` 施加。

## 2. 平台能力一览

| 能力 | 状态 | 指标 |
|---|---|---|
| MuJoCo 直接肌肉控制 | ✅ | 单段 **1.83°**,批量 **119/119 @ 2.02°** |
| Isaac Lab 对照版 | ✅ 凭证 | 1.78°(`debug/walk_muscle_demo.py`) |
| 肌肉可视化 | ✅ | 284 线灰→红 + 半径/透明度随激活;离屏视频 + 实时 viewer |
| 激活分析 | ✅ | 热图 + 逐帧动画(`plot_activation.py`) |
| 偏瘫实验 | 🔶 第一步 | 患侧 F0 80/60/40% 强度扫描 |
| Tendon 验证 | ✅ | waypoint vs spatial-tendon → **KEEP_WAYPOINT** |

## 3. 各阶段成果

### 3.1 Isaac Lab 版(迁移前,保留作凭证)
- 单段 **1.78°**、扭矩匹配 0.72;批量 119/119 @ 3.01°;
- 4 个关键 bug 已修复:`quat_to_exp_map`(euler 2π 环绕)、关节顺序重排、`os._exit(0)`、`simulation_app.close()` 挂起。

### 3.2 MuJoCo 迁移(核心)
- `protomotions/utils/direct_muscle_mujoco.py`:与 Isaac 版同接口,CPU 仿真;
- lbfgs+50 后单段 **1.83°**(Isaac 1.78°)、批量 **2.02°**(优于 Isaac 3.01°);
- 关键坑:
  1. `dof_armature=0.03` 必须设置,否则自由关节 QACC 爆炸;
  2. 优化必须 `lbfgs + 50`(ls 最小范数解被裁剪后失效);
  3. 力矩臂口径 τ=f·r:`ten_velocity` 取负与生产 `JtA` 对齐;
  4. bio.xml 自带 stiffness/damping 需置零(Isaac 执行器 stiffness=0)。

### 3.3 可视化体系
- 视频渲染:网格地面 + 阴影 + 半透明骨骼 + 全身视角;**60fps 由 240Hz 物理子步插值**;
- 肌肉线:`mjv_connector` 胶囊,颜色灰(0)→红(1),半径/透明度随激活;
- 激活图:整段热图 + 逐帧动画 GIF(9 解剖区域分组);
- 实时 viewer:`mujoco_demo/viewer_demo/view_demo_muscles.py`(glfw + PyOpenGL 自绘);
  - 黑屏坑:MuJoCo Renderer 渲染后需 `glfw.make_context_current` 恢复窗口上下文;
  - 已知限制:每子步优化导致 ~0.2x 倍速(可 `--max-iter 20 --muscle-stride 3` 缓解)。

### 3.4 偏瘫平台第一步:患侧 F0 强度实验
`--affected-side L --strength-sweep`(100/80/60/40%):

| 强度 | 全身(°) | 患侧(°) | 健侧(°) | τ匹配 |
|---|---|---|---|---|
| 100% | 1.83 | 1.56 | 0.96 | 0.563 |
| 80% | 1.81 | 1.48 | 0.94 | 0.567 |
| 60% | 1.83 | 1.44 | 0.90 | 0.587 |
| 40% | 1.79 | 1.41 | 0.83 | 0.601 |

**结论**:逐帧全知优化自动提高患侧激活补偿 f0 损失,肌力降到 40% 跟踪几乎不变。
单纯"肌力减弱"不足以模拟偏瘫步态;下一步需**痉挛(速度相关阻力)、激活上限(神经驱动受限)、足下垂**等机制。另:基线本身存在 L/R 不对称(PD-only 1.09° vs 0.81°),与肌肉无关。

## 4. 下一步(偏瘫平台 Phase 1)

1. 痉挛(spasticity):患侧速度相关被动阻力;
2. 激活上限:患侧 `a ≤ a_max`(神经驱动受限);
3. 足下垂:踝背屈肌弱化 / 激活抑制;
4. 双色可视化:患侧 / 健侧肌肉不同色系(viewer + 视频)。

## 5. 复现命令

```bash
# MuJoCo 单段 / 批量
PYTHONPATH=. python debug/walk_muscle_demo_mujoco.py --motion data/cmu_bio_npy/009/09_12.npy
PYTHONPATH=. python debug/walk_muscle_demo_mujoco.py --motion "data/cmu_bio_npy/*/*.npy" --batch

# 10 秒肌肉行走视频 (60fps, 网格地面, 全身)
PYTHONPATH=. python debug/walk_muscle_demo_mujoco.py --motion data/cmu_bio_npy/009/09_12.npy \
    --max-frames 300 --render-video output/mj_muscle_walk.mp4 --render-width 1280 --render-height 960

# 284 激活图 (热图 + 动画)
PYTHONPATH=. python debug/walk_muscle_demo_mujoco.py --motion data/cmu_bio_npy/009/09_12.npy \
    --save-npz output/mj_traj.npz
PYTHONPATH=. python debug/plot_activation.py --npz output/mj_traj.npz \
    --png output/activation_heatmap.png --gif output/activation_anim.gif

# 实时交互 viewer
PYTHONPATH=. python mujoco_demo/viewer_demo/view_demo_muscles.py --loop

# 偏瘫患侧强度扫描
PYTHONPATH=. python debug/walk_muscle_demo_mujoco.py --motion data/cmu_bio_npy/009/09_12.npy \
    --affected-side L --strength-sweep
```

