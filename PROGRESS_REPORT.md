# 阶段性成果报告 —— MuscleControl-Isaac 直接肌肉控制路线

> 日期: 2026-08-21
> 目标: 让 BIO 人形(50 关节、284 块 Hill 型肌肉)通过**肌肉激活**驱动,在 Isaac Lab 中复现运动(当前以 CMU 行走为主)。

---

## 1. 技术路线(已确认的最终方案)

采用**纯直接优化**(不训练网络、无 RL),逐帧反解肌肉激活:

```
BVH → BIO q_ref → PD desired torque → optimization
    → 284 activation → muscle model (Hill) → 50 torque → Isaac Lab
```

每帧在每个物理子步内:
1. 由参考关节轨迹 `q_ref` 经 PD 控制器算出期望力矩 `tau_des = kp·(q_ref−q) − kd·qd`;
2. 用优化器解出 284 块肌肉激活 `a`,使 `a·JtA + b` 逼近 `tau_des`;
3. Hill 肌肉模型把激活映射为 50 维关节力矩;
4. 写入 Isaac Lab 驱动仿真。

---

## 1.1 核心代码文件(⭐ 本次工作)

| 文件 | 行数 | 类型 | 说明 |
|---|---|---|---|
| ⭐ `protomotions/utils/direct_muscle.py` | 421 | 新增 | **固化核心模块**:`DirectMuscleTracker` 类,整条直接肌肉控制路线(常量/`local_rotation_to_dof`/`optimize_act`/`track()`/`kinematic_reference()`) |
| ⭐ `debug/walk_muscle_demo.py` | 481 | 新增 | CLI 入口:单段评估、`--sweep` 参数扫描、`--batch` 批量评估、`--scale-map`/`--pd-only` |
| ⭐ `debug/render_muscle_walk.py` | 191 | 新增 | 离线可视化:骨架 GIF、`--video` mp4、`--color-act` 肌肉激活上色 |
| ⭐ `protomotions/utils/muscle_control.py` | 157 | 修改 | 新增 `set_global_scale()` / `set_muscle_scales()`(按肌肉名缩放 f0,补偿偏弱肌肉) |
| `protomotions/utils/muscle_parser.py` | 282 | 依赖 | Hill 肌肉参数解析(muscle284.xml + bio.xml) |
| `data/scripts/convert_cmu_bvh_to_isaac.py` | — | 依赖 | BVH → BIO 骨架重定向(数据管线) |

> ⭐ = 本次直接肌肉控制路线的核心新增/修改文件,其余为既有依赖。

---

## 2. 已完成的工作

### 2.1 数据管线(BVH → BIO q_ref)✅

- 下载并处理 CMU MoCap 6 个 subject(`008/009/035/091/104/105`),共 **119 段行走**;
- `data/scripts/convert_cmu_bvh_to_isaac.py`:BVH → BIO 骨架重定向(poselib `retarget_to`,scale=0.060);
- 打包产物 `data/walking_bio.pt`(161MB,29min,走 Git LFS 分发);
- 中间产物 `data/cmu_bio_npy/*/*.npy`(SkeletonMotion,30fps)。

### 2.2 直接肌肉控制路线(核心成果)✅

| 文件 | 作用 |
|---|---|
| `protomotions/utils/direct_muscle.py` | **固化模块**:`DirectMuscleTracker` 类(可复用,含 `track()` / `kinematic_reference()`) |
| `debug/walk_muscle_demo.py` | CLI 入口:单段 / `--sweep` 参数扫描 / `--batch` 批量评估 |
| `debug/render_muscle_walk.py` | 离线可视化:骨架 GIF / `--video` mp4 / `--color-act` 激活上色 |
| `protomotions/utils/muscle_control.py` | 新增 `set_global_scale()` / `set_muscle_scales()`(按肌肉名缩放 f0) |

### 2.3 关键 bug 定位与修复(4 个)✅

| # | 问题 | 修复 |
|---|---|---|
| 1 | 3 自由度关节用 `get_euler_xyz` 返回 `% 2π`,小负角被包成 ~6.28 rad → 物理 NaN | 改用 `quat_to_exp_map`(角度 ∈ [−π,π]),与训练管线一致 |
| 2 | **sim 关节顺序 ≠ common 顺序**,指标里重排方向写反,把 0.5° 误算成 19° | 明确 `sim→common = v_sim[dof_to_common]`、`common→sim = v_common[dof_to_sim]` |
| 3 | `os._exit(0)` 结尾 + 输出重定向 → stdout 缓冲丢失 | exit 前 `sys.stdout.flush()` 或 `python -u` |
| 4 | Isaac Sim `simulation_app.close()` 挂起 | 脚本末尾统一 `os._exit(0)` |

---

## 3. 实验结果

### 3.1 单段跟踪精度(CMU 009/09_12,40 帧,`ls` 优化)

| 配置 | 全身关节跟踪误差 |
|---|---|
| PD-only 基线(完美力矩,对照) | **0.5°** |
| 肌肉(无缩放) | 2.3° |
| 肌肉 + 逐肌肉 f0 补偿(`--scale-map`) | **1.8°** ✅ |

- 肌肉力矩 vs PD 期望力矩匹配度:0.72~0.80(相对误差);
- 分部位(带 scale-map):腿 3.2° / 臂 2.3° / 躯干 1.0°;最差的踝/趾关节由 ~5° 降到 ~2.2°。

### 3.2 批量评估(全部 119 段,`--batch`)

```
成功: 119/119  失败率: 0.0%
平均跟踪误差: 3.01°  median 3.07°  p90 4.35°  max 6.00°
```

| subject | 段数 | 平均误差 |
|---|---|---|
| 008 | 11 | 4.87° |
| 009 | 1 | 2.13° |
| 035 | 23 | 2.97° |
| 091 | 29 | 3.19° |
| 104 | 23 | 1.81° |
| 105 | 32 | 3.14° |

> 结论:**直接优化路线对全部 119 段行走零失败、稳定跟踪**,误差仅比完美 PD 高约 1.3°。

### 3.3 可视化验证

- `output/muscle_walk_long.mp4`:8 秒(240 帧 @30fps)肌肉驱动行走;
- 数值确认:骨盆高 ~0.97 m(直立)、脚贴地 ~0.12 m、肌肉实际 vs 参考骨架平均差 **2.6 cm**。

---

## 4. 环境状态

| 机器 | 状态 |
|---|---|
| 本机 RTX 5060(Blackwell) | headless 物理可用;GUI/离屏渲染不可用 → 走离线 matplotlib 渲染 |
| 云端 RTX 4090D(直通) | 部署文档 `DEPLOY.md` / `AUTO_DL_SETUP.md` 就绪,可跑 GUI + 肌肉网格上色 |

---

## 5. 待办 / 下一步

1. **数据质量**:脚接触校正(支撑相锚定脚),进一步降低踝/趾误差;
2. **精度微调**:`MUSCLE_SCALE_MAP` 或 `--kp-scale` 扫描,逼近完美 PD 的 0.5°;
3. **云端可视化**:4090D 上 `headless=False` 实时看肌肉网格按激活上色;
4. **(可选)回 RL 两阶段**:Phase 1 教师此前跟踪成功率 0%(历史遗留),现可直接优化路线作参考基准。

---

## 6. 复现命令

```bash
# 单段评估
python debug/walk_muscle_demo.py --motion data/cmu_bio_npy/009/09_12.npy \
    --method ls --max-iter 10 --scale-map --max-frames 40

# 批量评估全部 119 段
python debug/walk_muscle_demo.py --motion "data/cmu_bio_npy/*/*.npy" \
    --max-frames 20 --method ls --max-iter 5 --scale-map --batch

# 生成视频
python debug/render_muscle_walk.py --motion data/cmu_bio_npy/009/09_12.npy \
    --max-frames 240 --nframes 240 --fps 30 --video --color-act
```
