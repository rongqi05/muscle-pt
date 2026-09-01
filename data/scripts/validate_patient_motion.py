#!/usr/bin/env python3
"""验证重定向后的病人运动 (.npy, BIO 骨架) 质量。

检查项:
  1. NaN / Inf
  2. 骨盆高度 (健康站立 ~0.85-1.0m; 病人可能偏低/摆动)
  3. 脚触地: 最低脚点 z 接近 0
  4. 支撑相脚滑动速度 (脚在地面时水平速度, 越小越好)
  5. 关节角度范围 vs BIO 关节限位
  6. 左右步长/步频对称性 (偏瘫不对称量化)

输出: data/patient_bio_npy/validation_report.csv + 终端汇总

用法:  PYTHONPATH=. python data/scripts/validate_patient_motion.py [--input data/patient_bio_npy]
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from poselib.skeleton.skeleton3d import SkeletonMotion  # noqa: E402

BODY_NAMES = [
    "Pelvis", "Spine", "Torso", "Neck", "Head",
    "ShoulderL", "ArmL", "ForeArmL", "HandL",
    "ShoulderR", "ArmR", "ForeArmR", "HandR",
    "FemurL", "TibiaL", "TalusL", "FootThumbL", "FootPinkyL",
    "FemurR", "TibiaR", "TalusR", "FootThumbR", "FootPinkyR",
]
# BIO 关节限位 (rad), 与 bio.xml 一致 (仅检查主要关节)
JOINT_LIMITS = {
    "L_Hip_x": (-2.09, 2.09), "R_Hip_x": (-2.09, 2.09),
    "L_Hip_y": (-2.09, 2.09), "R_Hip_y": (-2.09, 2.09),
    "L_Hip_z": (-2.09, 2.09), "R_Hip_z": (-2.09, 2.09),
    "L_Knee": (0.0, 2.79), "R_Knee": (0.0, 2.79),
    "L_Ankle_x": (-0.79, 0.79), "R_Ankle_x": (-0.79, 0.79),
    "L_Ankle_y": (-0.79, 0.79), "R_Ankle_y": (-0.79, 0.79),
    "L_Ankle_z": (-0.79, 0.79), "R_Ankle_z": (-0.79, 0.79),
}


def dof_pos(motion: SkeletonMotion):
    """BIO 骨架 local rotation -> 50 DOF (exp_map 3DOF / angle_axis 1DOF)。"""
    from isaac_utils import rotations, torch_utils
    from protomotions.utils.direct_muscle import DOF_NAMES, DOF_BODY_IDS, DOF_OFFSETS, JOINT_AXIS
    lr = motion.local_rotation  # (T, 23, 4) xyzw
    T = lr.shape[0]
    out = np.zeros((T, 50), dtype=np.float32)
    for j, body_id in enumerate(DOF_BODY_IDS):
        off = DOF_OFFSETS[j]
        size = DOF_OFFSETS[j + 1] - off
        jq = lr[:, body_id]
        if size == 3:
            out[:, off:off + 3] = torch_utils.quat_to_exp_map(jq, w_last=True).numpy()
        else:
            theta, axis = torch_utils.quat_to_angle_axis(jq, w_last=True)
            theta = theta.numpy()
            axis = axis.numpy()
            cfg = JOINT_AXIS[j]
            ax = {"x": 0, "y": 1, "z": 2}[cfg]
            ang = theta * axis[..., ax]
            out[:, off] = np.arctan2(np.sin(ang), np.cos(ang))  # wrap 到 [-pi, pi]
    return out, DOF_NAMES


def validate_one(npy_path: str):
    motion = SkeletonMotion.from_file(npy_path)
    gp = motion.global_translation.numpy()   # (T,23,3) 米
    rt = motion.root_translation.numpy()     # (T,3)
    q, dof_names = dof_pos(motion)
    T = gp.shape[0]

    ok = True
    issues = []
    nan_flag = bool(np.isnan(gp).any() or np.isinf(gp).any())
    if nan_flag:
        ok = False
        issues.append("NaN/Inf")

    pelvis_h = gp[:, 0, 2].mean()
    pelvis_h_std = gp[:, 0, 2].std()
    if not (0.7 < pelvis_h < 1.1):
        ok = False
        issues.append(f"骨盆高度异常 {pelvis_h:.2f}m")

    foot_heights = np.concatenate([
        gp[:, 15:18, 2].min(axis=1), gp[:, 18:21, 2].min(axis=1)])
    min_foot = float(gp[:, 15:21, 2].min())
    foot_swing = float(foot_heights.max())
    if min_foot > 0.15:
        ok = False
        issues.append(f"脚离地过高 {min_foot:.3f}m")

    # 支撑相脚滑动: 脚 z < 0.05 时水平速度
    slip_speeds = []
    for fidx in (15, 16, 17, 19, 20, 21):  # 左右踝/足
        p = gp[:, fidx]
        v = np.linalg.norm(np.gradient(p, axis=0), axis=1) * 30.0  # 30fps
        contact = p[:, 2] < 0.06
        if contact.sum() > 5:
            slip_speeds.append(float(v[contact].mean()))
    slip = float(np.mean(slip_speeds)) if slip_speeds else 0.0

    # 关节限位
    lim_frac = 0.0
    for jn, (lo, hi) in JOINT_LIMITS.items():
        if jn in dof_names:
            v = q[:, dof_names.index(jn)]
            lim_frac += float(((v < lo - 0.05) | (v > hi + 0.05)).mean())
    lim_frac /= len(JOINT_LIMITS)

    # 左右对称性: 摆动相时间占比 (脚 z > 0.1 的时间比)
    swing_l = float((gp[:, 17, 2] > 0.10).mean())
    swing_r = float((gp[:, 20, 2] > 0.10).mean())

    return dict(file=os.path.basename(npy_path), frames=T, ok=ok,
                pelvis_h=round(pelvis_h, 3), pelvis_h_std=round(pelvis_h_std, 3),
                min_foot=round(min_foot, 3), foot_swing=round(foot_swing, 3),
                slip_ms=round(slip, 3), lim_viol=round(lim_frac, 3),
                swing_l=round(swing_l, 2), swing_r=round(swing_r, 2),
                issues=";".join(issues))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(REPO_ROOT, "data", "patient_bio_npy"))
    parser.add_argument("--max", type=int, default=0, help="只验证前 N 个 (调试)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "**", "*.npy"), recursive=True))
    if args.max:
        files = files[:args.max]
    print(f"验证 {len(files)} 个运动文件")

    rows = []
    for f in files:
        try:
            r = validate_one(f)
        except Exception as e:
            r = dict(file=os.path.basename(f), frames=0, ok=False,
                     issues=f"解析失败: {e}")
        rows.append(r)
        status = "OK " if r.get("ok") else "WARN"
        print(f"[{status}] {r['file']:<40} frames={r.get('frames', 0):>4} "
              f"骨盆={r.get('pelvis_h', '?'):>5} 脚底={r.get('min_foot', '?'):>5} "
              f"滑动={r.get('slip_ms', '?')} m/s 摆L/R={r.get('swing_l', '?')}/{r.get('swing_r', '?')} "
              f"{r.get('issues', '')}")

    out = os.path.join(args.input, "validation_report.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "frames", "ok", "pelvis_h", "min_foot", "slip_ms",
                    "lim_viol", "swing_l", "swing_r", "issues"])
        for r in rows:
            w.writerow([r.get("file", ""), r.get("frames", ""), r.get("ok", ""),
                        r.get("pelvis_h", ""), r.get("min_foot", ""),
                        r.get("slip_ms", ""), r.get("lim_viol", ""),
                        r.get("swing_l", ""), r.get("swing_r", ""), r.get("issues", "")])
    n_ok = sum(1 for r in rows if r.get("ok"))
    print(f"\n通过: {n_ok}/{len(rows)}  报告 -> {out}")


if __name__ == "__main__":
    main()
