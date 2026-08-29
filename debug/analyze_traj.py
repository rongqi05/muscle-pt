#!/usr/bin/env python3
"""行走 demo 轨迹数据分析 (输入 export_traj.py 导出的 CSV 目录)。

输出 analysis/ 目录:
  tracking_error.png        关节跟踪误差 (top10 条形图 + 主要关节时间曲线)
  symmetry.png              左右对称性 (激活均值曲线 + 左右关节误差差)
  activation_heatmap.png    284 肌肉激活热图 (左右分界中线)
  torque_match.png          力矩匹配度时间曲线
  report.txt                指标汇总

用法:
    PYTHONPATH=. python debug/export_traj.py --npz output/mj_traj.npz --out output/export
    PYTHONPATH=. python debug/analyze_traj.py --dir output/export
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """读 export_traj.py 的 CSV -> (数据 (T,N), 列名 list)。首列 frame 被丢弃。"""
    with open(path) as f:
        header = f.readline().strip().split(",")
    cols = header[1:]
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 1:], cols


def load(dirpath):
    out = {}
    for name in ("q", "q_ref", "tau_des", "tau_muscle", "activation"):
        data, cols = load_csv(os.path.join(dirpath, f"{name}.csv"))
        out[name] = (data, cols)
    return out


def joint_group(name):
    if any(k in name for k in ("Hip", "Knee", "Ankle", "Toe")):
        return "leg"
    if any(k in name for k in ("Shoulder", "Elbow", "ForeArm", "Hand")):
        return "arm"
    return "trunk"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="output/export", help="CSV 目录")
    args = parser.parse_args()

    d = load(args.dir)
    outdir = os.path.join(args.dir, "analysis")
    os.makedirs(outdir, exist_ok=True)
    q, qcols = d["q"]
    qref, _ = d["q_ref"]
    taud, _ = d["tau_des"]
    taum, _ = d["tau_muscle"]
    a, acols = d["activation"]

    # ---- 误差 ----
    err = np.degrees(np.abs(q - qref))          # (T, 50) 度
    err_mean = err.mean(axis=0)                  # 每关节均值
    lines = []
    lines.append("=" * 60)
    lines.append("行走轨迹分析报告")
    lines.append(f"帧数: {q.shape[0]}  关节: {q.shape[1]}  肌肉: {a.shape[1]}")
    lines.append("-" * 60)
    lines.append(f"全身跟踪误差 (deg): {err.mean():.3f}")
    for grp in ("leg", "trunk", "arm"):
        mask = [joint_group(c) == grp for c in qcols]
        lines.append(f"  {grp:<6}: {err[:, mask].mean():.3f}")
    # 左右不对称 (对应关节: L_xxx <-> R_xxx)
    lmask = [c.startswith("L_") for c in qcols]
    rmask = [c.startswith("R_") for c in qcols]
    almask = [c.startswith("L_") for c in acols]
    armask = [c.startswith("R_") for c in acols]
    lerr = err[:, lmask].mean()
    rerr = err[:, rmask].mean()
    lines.append(f"左侧误差 {lerr:.3f}° vs 右侧 {rerr:.3f}°  (不对称 = {lerr - rerr:+.3f}°)")
    # 激活对称性
    amean = a.mean(axis=0)
    lact = amean[almask].mean()
    ract = amean[armask].mean()
    lines.append(f"左侧激活均值 {lact:.3f} vs 右侧 {ract:.3f}  (差 = {lact - ract:+.3f})")
    # 力矩匹配
    match = 1 - np.linalg.norm(taum - taud, axis=1) / (np.linalg.norm(taud, axis=1) + 1e-9)
    lines.append(f"力矩匹配度: 均值 {match.mean():.3f}  最小 {match.min():.3f}")
    lines.append("=" * 60)

    # ---- 图 1: 跟踪误差 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    top_idx = np.argsort(err_mean)[-10:]
    top_names = [qcols[i] for i in top_idx]
    axes[0].barh(top_names, err_mean[top_idx], color="#d62728")
    axes[0].set_xlabel("mean tracking error (deg)"); axes[0].set_title("Top-10 worst joints")
    for jn in ("L_Knee", "R_Knee", "L_Hip_x", "R_Hip_x", "L_Ankle_x", "R_Ankle_x"):
        if jn in qcols:
            axes[1].plot(err[:, qcols.index(jn)], label=jn, lw=1.2)
    axes[1].set_xlabel("frame"); axes[1].set_ylabel("error (deg)")
    axes[1].legend(ncol=2, fontsize=8); axes[1].set_title("Lower-limb joint error over time")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "tracking_error.png"), dpi=110)
    plt.close(fig)

    # ---- 图 2: 对称性 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].plot(a[:, almask].mean(axis=1), label="left", color="#1f77b4", lw=1.5)
    axes[0].plot(a[:, armask].mean(axis=1), label="right", color="#d62728", lw=1.5)
    axes[0].set_xlabel("frame"); axes[0].set_ylabel("mean activation")
    axes[0].legend(); axes[0].set_title("Left vs right mean activation")
    pair_err = []
    for lc in [c for c in qcols if c.startswith("L_")]:
        rc = lc.replace("L_", "R_")
        if rc in qcols:
            pair_err.append(err[:, qcols.index(lc)].mean() - err[:, qcols.index(rc)].mean())
    pair_err = np.array(pair_err)
    axes[1].bar(range(len(pair_err)), pair_err,
                color=["#1f77b4" if v >= 0 else "#d62728" for v in pair_err])
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("joint pair index"); axes[1].set_ylabel("L err - R err (deg)")
    axes[1].set_title("Left-right joint error difference (positive = left worse)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "symmetry.png"), dpi=110)
    plt.close(fig)

    # ---- 图 3: 激活热图 ----
    amat = a.T                                     # (284, T)
    idx = sorted(range(amat.shape[0]), key=lambda i: acols[i])
    amat = amat[idx]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(amat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    lcount = sum(1 for c in acols if c.startswith("L_"))
    ax.axhline(lcount - 0.5, color="white", lw=1.5)
    ax.set_xlabel("frame"); ax.set_ylabel("muscle (sorted)")
    ax.set_title("284 muscle activation (white line = L/R boundary)")
    fig.colorbar(im, ax=ax, fraction=0.02); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "activation_heatmap.png"), dpi=110)
    plt.close(fig)

    # ---- 图 4: 力矩匹配 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(match, lw=1.2, color="#2ca02c")
    axes[0].set_xlabel("frame"); axes[0].set_ylabel("torque match")
    axes[0].set_title("Torque match (1 = perfect)")
    axes[1].hist(match, bins=30, color="#2ca02c", alpha=0.8)
    axes[1].set_xlabel("torque match"); axes[1].set_title("分布")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "torque_match.png"), dpi=110)
    plt.close(fig)

    report = "\n".join(lines)
    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n图与报告已保存 -> {outdir}/")


if __name__ == "__main__":
    main()
