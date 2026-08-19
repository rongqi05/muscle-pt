#!/usr/bin/env python3
"""从 output/replay_traj.npz 渲染回放 GIF (纯 matplotlib, 不依赖 Isaac Sim)。

用法:
    python debug/render_replay.py --npz output/replay_traj.npz --gif output/replay.gif
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_bio_parents(body_names):
    tree = {
        "Pelvis": -1, "Spine": 0, "Torso": 1, "Neck": 2, "Head": 3,
        "ShoulderL": 2, "ArmL": 5, "ForeArmL": 6, "HandL": 7,
        "ShoulderR": 2, "ArmR": 9, "ForeArmR": 10, "HandR": 11,
        "FemurL": 0, "TibiaL": 13, "TalusL": 14, "FootThumbL": 15, "FootPinkyL": 15,
        "FemurR": 0, "TibiaR": 18, "TalusR": 19, "FootThumbR": 20, "FootPinkyR": 20,
    }
    return [tree.get(n, -1) for n in body_names]


def draw_skeleton(ax, gp, parents, color, lw=2.0, alpha=1.0):
    for j in range(1, len(parents)):
        p = int(parents[j])
        if p < 0 or p >= len(gp):
            continue
        ax.plot(
            [gp[p, 0], gp[j, 0]],
            [gp[p, 1], gp[j, 1]],
            [gp[p, 2], gp[j, 2]],
            color=color, linewidth=lw, alpha=alpha,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="output/replay_traj.npz")
    parser.add_argument("--gif", default="output/replay.gif")
    parser.add_argument("--frames", default="output/replay_frames")
    parser.add_argument("--nframes", type=int, default=30)
    parser.add_argument("--dur", type=int, default=120, help="每帧毫秒")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    d = np.load(args.npz)
    actual = d["actual_pos"]
    ref = d["ref_pos"]
    body_names = [str(b) for b in d["body_names"]]
    parents = get_bio_parents(body_names)

    T = actual.shape[0]
    has_ref = ref.shape[0] > 0
    print(f"实际轨迹: {actual.shape}  参考轨迹: {'有' if has_ref else '无'}")

    os.makedirs(args.frames, exist_ok=True)
    os.makedirs(os.path.dirname(args.gif), exist_ok=True)

    n_frames = min(T, args.nframes)
    idxs = np.linspace(0, T - 1, n_frames).astype(int)

    all_pos = actual if not has_ref else np.concatenate([actual, ref], axis=0)
    c = (all_pos.max(axis=(0, 1)) + all_pos.min(axis=(0, 1))) / 2.0
    r = float((all_pos.max(axis=(0, 1)) - all_pos.min(axis=(0, 1))).max()) * 0.6 + 0.5

    frames = []
    for k, i in enumerate(idxs):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        draw_skeleton(ax, actual[i], parents, "tab:blue", lw=2.2)
        if has_ref:
            draw_skeleton(ax, ref[i], parents, "tab:red", lw=1.6, alpha=0.7)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_title(f"step {int(i)}  蓝=策略 红=参考" if has_ref else f"step {int(i)}  策略")
        ax.view_init(elev=20, azim=-60)
        fpath = os.path.join(args.frames, f"frame_{k:03d}.png")
        fig.savefig(fpath, dpi=110)
        plt.close(fig)
        frames.append(Image.open(fpath).convert("RGB"))

    if frames:
        frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                       duration=args.dur, loop=0)
        print(f"GIF 已保存: {args.gif} ({len(frames)} 帧)")
        print(f"关键帧: {args.frames}/")


if __name__ == "__main__":
    main()
