#!/usr/bin/env python3
"""导出行走 demo 轨迹数据为 CSV (供 MATLAB/Python/Excel 分析)。

输入: walk_muscle_demo_mujoco.py --save-npz 的轨迹 npz
输出 (out 目录):
    q.csv / q_ref.csv          关节角度 (T x 50, 列=DOF_NAMES)
    tau_des.csv / tau_muscle.csv 期望/肌肉力矩 (T x 50, N·m)
    activation.csv             284 肌肉激活 (T x 284, 列=muscle_names)
    body_pos.csv               刚体世界位置 (T x 23x3)
    summary.json               汇总指标 (跟踪误差/扭矩匹配/激活统计)

用法:
    python debug/walk_muscle_demo_mujoco.py --motion data/cmu_bio_npy/009/09_12.npy \
        --max-frames 300 --save-npz output/mj_traj.npz
    PYTHONPATH=. python debug/export_traj.py --npz output/mj_traj.npz --out output/export
"""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from protomotions.utils.direct_muscle import DOF_NAMES, BODY_NAMES  # noqa: E402


def save_csv(path, data, columns):
    header = "frame," + ",".join(columns)
    rows = np.column_stack([np.arange(data.shape[0]), data])
    np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"  {os.path.basename(path)}: {data.shape} -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="output/mj_traj.npz")
    parser.add_argument("--out", default="output/export")
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    os.makedirs(args.out, exist_ok=True)
    print(f"输入: {args.npz} (字段: {list(d.files)})")

    q = d["q"]; q_ref = d["q_ref"]
    tau_des = d["tau_des"]; tau_muscle = d["tau_muscle"]
    a = d["a"]
    T = q.shape[0]

    save_csv(os.path.join(args.out, "q.csv"), q, DOF_NAMES)
    save_csv(os.path.join(args.out, "q_ref.csv"), q_ref, DOF_NAMES)
    save_csv(os.path.join(args.out, "tau_des.csv"), tau_des, DOF_NAMES)
    save_csv(os.path.join(args.out, "tau_muscle.csv"), tau_muscle, DOF_NAMES)

    muscle_names = [str(n) for n in d["muscle_names"]] if "muscle_names" in d.files \
        else [f"muscle_{i}" for i in range(a.shape[1])]
    save_csv(os.path.join(args.out, "activation.csv"), a, muscle_names)

    if "body_pos" in d.files:
        bp = d["body_pos"]  # (T,23,3)
        cols = [f"{b}_{ax}" for b in BODY_NAMES for ax in ("x", "y", "z")]
        save_csv(os.path.join(args.out, "body_pos.csv"), bp.reshape(T, -1), cols)

    # 汇总指标
    tracking_err = float(np.abs(q - q_ref).mean())
    denom = np.linalg.norm(tau_des, axis=-1) + 1e-6
    torque_match = float(np.mean(np.linalg.norm(tau_muscle - tau_des, axis=-1) / denom))
    summary = {
        "frames": T,
        "dof_count": len(DOF_NAMES),
        "muscle_count": a.shape[1],
        "tracking_err_rad": tracking_err,
        "tracking_err_deg": float(np.degrees(tracking_err)),
        "torque_match": torque_match,
        "activation_mean": float(a.mean()),
        "activation_sat_ratio": float(((a < 0.02) | (a > 0.98)).mean()),
        "nan_flag": bool(np.isnan(q).any() or np.isinf(q).any()),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n汇总指标:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n导出完成 -> {args.out}/")


if __name__ == "__main__":
    main()
