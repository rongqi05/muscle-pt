#!/usr/bin/env python3
"""尺度与骨架形态审计 (SCALE AND SKELETON MORPHOLOGY AUDIT)。

对 CMU BVH 源骨架与 BIO 骨架逐段测量, 判断现有 ~6x 缩放因子是:
  A. 合理的单位/全局缩放转换 (GLOBAL_SCALE_OR_UNIT_MISMATCH)
  还是
  B. 弥补骨架形态不兼容的 hack (SKELETON_PROPORTION_MISMATCH)

用法:
    python data/scripts/audit_skeleton_scale.py --bvh data/cmu_mocap/008/08_01.bvh \
        --bio protomotions/data/assets/mjcf/bio.xml [--motion data/cmu_bio_npy/008/08_01.npy]
"""
import argparse
import os
import sys

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data.scripts.lafan_utils import read_bvh, quat_fk  # noqa: E402
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion  # noqa: E402

# 对应骨段定义: (名称, CMU 关节A, CMU 关节B, BIO 关节A, BIO 关节B)
SEGMENTS = [
    ("pelvis->hip_L", "Hips", "LeftUpLeg", "Pelvis", "FemurL"),
    ("hip->knee_L", "LeftUpLeg", "LeftLeg", "FemurL", "TibiaL"),
    ("knee->ankle_L", "LeftLeg", "LeftFoot", "TibiaL", "TalusL"),
    ("ankle->toe_L", "LeftFoot", "LeftToeBase", "TalusL", "FootThumbL"),
    ("pelvis->hip_R", "Hips", "RightUpLeg", "Pelvis", "FemurR"),
    ("hip->knee_R", "RightUpLeg", "RightLeg", "FemurR", "TibiaR"),
    ("knee->ankle_R", "RightLeg", "RightFoot", "TibiaR", "TalusR"),
    ("ankle->toe_R", "RightFoot", "RightToeBase", "TalusR", "FootThumbR"),
    ("pelvis->torso", "Hips", "Spine1", "Pelvis", "Torso"),
    # 注意: CMU 的 Neck 是零长度占位关节 (offset=0), 实际颈部在 Neck1
    ("torso->neck", "Spine1", "Neck1", "Torso", "Neck"),
    ("neck->head", "Neck1", "Head", "Neck", "Head"),
    ("shoulder->elbow_L", "LeftShoulder", "LeftArm", "ShoulderL", "ArmL"),
    ("elbow->wrist_L", "LeftArm", "LeftForeArm", "ArmL", "ForeArmL"),
    ("wrist->hand_L", "LeftForeArm", "LeftHand", "ForeArmL", "HandL"),
    ("shoulder->elbow_R", "RightShoulder", "RightArm", "ShoulderR", "ArmR"),
    ("elbow->wrist_R", "RightArm", "RightForeArm", "ArmR", "ForeArmR"),
    ("wrist->hand_R", "RightForeArm", "RightHand", "ForeArmR", "HandR"),
]


def cmu_global_positions(anim):
    """CMU 骨架 T-pose (单位旋转) 全局关节位置, 单位 cm。"""
    n = len(anim.bones)
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))  # wxyz 单位
    _, gpos = quat_fk(q[np.newaxis, ...], anim.offsets[np.newaxis, ...], anim.parents)
    return anim.bones, gpos[0]  # (J, 3) cm


def bio_global_positions(bio_xml):
    """BIO 骨架 zero-pose 全局关节位置, 单位 m。"""
    tree = SkeletonTree.from_mjcf(bio_xml)
    st = SkeletonState.zero_pose(tree)
    return list(tree.node_names), st.global_translation.numpy()


def main():
    ap = argparse.ArgumentParser(description="尺度与骨架形态审计")
    ap.add_argument("--bvh", required=True, help="CMU BVH 文件")
    ap.add_argument("--bio", required=True, help="BIO mjcf 文件")
    ap.add_argument("--motion", default=None, help="(可选) 现有 scale 重定向结果的 .npy, 用于质量审计")
    args = ap.parse_args()

    anim = read_bvh(args.bvh)
    cmu_names, cmu_gpos = cmu_global_positions(anim)
    bio_names, bio_gpos = bio_global_positions(args.bio)

    cmu_idx = {n: i for i, n in enumerate(cmu_names)}
    bio_idx = {n: i for i, n in enumerate(bio_names)}

    print("=" * 78)
    print("尺度与骨架形态审计报告")
    print("=" * 78)
    print(f"源 BVH: {args.bvh}")
    print(f"目标 BIO: {args.bio}")
    print(f"源关节数: {len(cmu_names)}, 目标关节数: {len(bio_names)}")

    # 1. 源骨架测量 (原生单位 cm)
    print("\n[1] 源骨架段长 (CMU, 原生单位 cm)")
    print(f"  {'段':<18}{'BVH(cm)':>10}")
    src_len = {}
    for seg, ca, cb, ba, bb in SEGMENTS:
        if ca not in cmu_idx or cb not in cmu_idx:
            print(f"  {seg:<18}{'N/A':>10} (缺关节 {ca}/{cb})")
            continue
        d = np.linalg.norm(cmu_gpos[cmu_idx[ca]] - cmu_gpos[cmu_idx[cb]])
        src_len[seg] = d
        print(f"  {seg:<18}{d:>10.3f}")
    # 源骨架总高
    src_height = cmu_gpos[:, 1].max() - cmu_gpos[:, 1].min()
    print(f"  {'总骨架高度(y轴)':<18}{src_height:>10.3f}")

    # 2. BIO 骨架测量 (m)
    print("\n[2] BIO 骨架段长 (m)")
    print(f"  {'段':<18}{'BIO(m)':>10}")
    bio_len = {}
    for seg, ca, cb, ba, bb in SEGMENTS:
        if ba not in bio_idx or bb not in bio_idx:
            print(f"  {seg:<18}{'N/A':>10} (缺关节 {ba}/{bb})")
            continue
        d = np.linalg.norm(bio_gpos[bio_idx[ba]] - bio_gpos[bio_idx[bb]])
        bio_len[seg] = d
        print(f"  {seg:<18}{d:>10.3f}")
    bio_height = bio_gpos[:, 1].max() - bio_gpos[:, 1].min()
    print(f"  {'总骨架高度(y轴)':<18}{bio_height:>10.3f}")

    # 3. 段比例表
    print("\n[3] 段比例 (BIO/BVH)")
    print(f"  {'段':<18}{'BVH':>8}{'BIO':>8}{'ratio':>10}")
    ratios = []
    ratio_by_seg = {}
    common = set(src_len) & set(bio_len)
    for seg, ca, cb, ba, bb in SEGMENTS:
        if seg not in common:
            continue
        # 跳过源长度为零的占位段 (无法计算比例)
        if src_len[seg] < 1e-6:
            print(f"  {seg:<18}{src_len[seg]:>8.3f}{bio_len[seg]:>8.3f}{'N/A(源长度0)':>10}")
            continue
        r = bio_len[seg] / max(src_len[seg], 1e-9)
        ratios.append(r)
        ratio_by_seg[seg] = r
        print(f"  {seg:<18}{src_len[seg]:>8.3f}{bio_len[seg]:>8.3f}{r:>10.3f}")

    ratios = np.array(ratios)
    mean_r = ratios.mean()
    median_r = np.median(ratios)
    std_r = ratios.std()
    cv = std_r / mean_r if mean_r > 0 else 0
    print(f"\n  统计: mean={mean_r:.3f}  median={median_r:.3f}  std={std_r:.3f}  "
          f"CV(变异系数)={cv:.3f}")

    # 长骨段 (腿+臂主体, 决定姿态与步态) 子集统计
    LONG_BONE_SEGS = ["hip->knee_L", "knee->ankle_L", "hip->knee_R", "knee->ankle_R",
                      "shoulder->elbow_L", "elbow->wrist_L", "shoulder->elbow_R", "elbow->wrist_R"]
    long_ratios = np.array([ratio_by_seg[s] for s in LONG_BONE_SEGS if s in ratio_by_seg])
    if len(long_ratios) > 0:
        lcv = long_ratios.std() / long_ratios.mean()
        print(f"  长骨段(腿+臂)统计: n={len(long_ratios)} mean={long_ratios.mean():.3f} "
              f"median={np.median(long_ratios):.3f} std={long_ratios.std():.3f} CV={lcv:.3f}")

    # 4. 分类
    print("\n[4] 分类")
    if cv < 0.15:
        cls = "GLOBAL_SCALE_OR_UNIT_MISMATCH"
        print(f"  -> {cls} (段比例一致性高, CV={cv:.3f} < 0.15, 全局缩放合理)")
    elif cv < 0.30:
        cls = "MIXED"
        print(f"  -> {cls} (段比例中等分散, CV={cv:.3f})")
    else:
        cls = "SKELETON_PROPORTION_MISMATCH"
        print(f"  -> {cls} (段比例分散, CV={cv:.3f} >= 0.30, 全局缩放不足)")
    # 差异最大的段
    dev = np.abs(ratios - mean_r)
    top = np.argsort(dev)[::-1][:3]
    print("  偏离均值最大的段:")
    seg_names = [s[0] for s in SEGMENTS if s[0] in common]
    for i in top:
        print(f"    {seg_names[i]}: ratio={ratios[i]:.3f} (偏离 {dev[i]:.3f})")

    # 5. 根运动尺度审计
    print("\n[5] 根运动尺度审计")
    root = anim.pos[:, 0, :]  # (T, 3) cm, 前进方向 z
    T = root.shape[0]
    dur = T / anim.fps
    disp = np.linalg.norm(root[-1, :] - root[0, :])  # 3D
    fwd_disp = root[-1, 2] - root[0, 2]  # z 前进
    mean_speed_cm = abs(fwd_disp) / dur
    print(f"  总前进位移(z): {fwd_disp:.2f} cm")
    print(f"  时长: {dur:.2f} s")
    print(f"  平均行走速度(源, 未缩放): {mean_speed_cm:.2f} cm/s = {mean_speed_cm*0.01:.3f} m/s")
    # 峰值速度 (帧间)
    v = np.abs(np.diff(root[:, 2])) / (1.0 / anim.fps)
    print(f"  峰值行走速度(源): {v.max():.2f} cm/s = {v.max()*0.01:.3f} m/s")

    # 步频估计: 骨盆高度 y 的 FFT 主频
    y = root[:, 1] - root[:, 1].mean()
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(T, d=1.0 / anim.fps)
    valid = (freqs > 0.3) & (freqs < 3.0)
    if valid.any():
        step_freq = freqs[valid][np.argmax(fft[valid])]
        stride = abs(mean_speed_cm) / max(step_freq, 1e-6)  # cm
        print(f"  步频(骨盆y主频): {step_freq:.2f} Hz")
        print(f"  步幅(速度/步频): {stride:.1f} cm")
    else:
        step_freq = stride = None

    # 缩放后速度 (scale 已含 cm->m 单位换算, 见第 3 节 ratio = BIO(m)/BVH(cm))
    print(f"\n  缩放因子候选: mean ratio = {mean_r:.3f} (median {median_r:.3f})")
    # scale 语义: 源(cm) × scale = 目标(m), 故速度换算无需再乘 0.01
    print(f"  缩放后平均速度: {mean_speed_cm * mean_r:.3f} m/s (mean) / {mean_speed_cm * median_r:.3f} m/s (median)")
    if step_freq is not None:
        print(f"  缩放后步幅: {stride * mean_r:.3f} m")
    print(f"  正常成人行走速度范围: ~1.0-1.6 m/s (慢走 0.5-0.8 m/s, 快走 1.6-2.0 m/s)")

    # 6. 重定向质量审计 (可选, 需要现有转换结果)
    if args.motion and os.path.exists(args.motion):
        print("\n[6] 重定向质量审计 (现有 scale 重定向结果)")
        m = SkeletonMotion.from_file(args.motion)
        names = list(m.skeleton_tree.node_names)
        idx = {n: i for i, n in enumerate(names)}
        gp = m.global_translation.numpy()
        # 脚高
        for b in ["FootThumbL", "FootThumbR", "TalusL", "TalusR"]:
            h = gp[:, idx[b], 2]
            print(f"  {b} 高度: min={h.min():.3f} mean={h.mean():.3f}")
        min_z = gp[..., 2].min()
        print(f"  最低点 z: {min_z:.3f} (穿透 = {abs(min(min_z, 0)):.3f} m)")
        # 根高
        pel = gp[:, idx["Pelvis"], 2]
        print(f"  骨盆高度: min={pel.min():.3f} mean={pel.mean():.3f} max={pel.max():.3f}")
        # 支撑脚滑动 (脚速度)
        fps = m.fps
        for b in ["TalusL", "TalusR"]:
            fv = np.linalg.norm(np.diff(gp[:, idx[b], :], axis=0), axis=1) * fps
            # 支撑相 = 脚接近地面
            h = gp[:, idx[b], 2]
            stance = h[1:] < (h.min() + 0.02)
            if stance.any():
                print(f"  {b} 支撑相滑动速度: mean={fv[stance].mean():.3f} m/s max={fv[stance].max():.3f}")

    print("\n" + "=" * 78)
    print("结论摘要")
    print(f"  分类: {cls}")
    print(f"  实测全局缩放(均值比例): {mean_r:.3f} (中位数 {median_r:.3f})")
    print("=" * 78)


if __name__ == "__main__":
    main()
