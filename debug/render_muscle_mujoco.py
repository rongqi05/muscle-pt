#!/usr/bin/env python3
"""MuJoCo 版肌肉驱动行走的离线渲染 (纯 matplotlib, 无需 Isaac Lab / GPU)。

输入: walk_muscle_demo_mujoco.py --save-npz 保存的轨迹 (actual=body_pos, ref, a)。
输出: 骨架 GIF / mp4, 可选按肌肉激活给骨骼上色。

用法:
    # 先保存轨迹
    python debug/walk_muscle_demo_mujoco.py --motion data/cmu_bio_npy/009/09_12.npy \
        --save-npz output/mj_traj.npz

    # 再渲染 (蓝=肌肉实际, 红=参考)
    python debug/render_muscle_mujoco.py --npz output/mj_traj.npz --gif output/mj_walk.gif --video
"""

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from protomotions.utils.direct_muscle_mujoco import DirectMuscleTrackerMujoco, BODY_NAMES  # noqa: E402

BIO_PARENTS = [-1, 0, 1, 2, 3,
               2, 5, 6, 7,
               2, 9, 10, 11,
               0, 13, 14, 15, 15,
               0, 18, 19, 20, 20]


def draw_skeleton(ax, gp, parents, color, lw=2.0, alpha=1.0, per_bone_colors=None):
    for j in range(1, len(parents)):
        p = int(parents[j])
        if p < 0 or p >= len(gp):
            continue
        c = color if per_bone_colors is None else tuple(per_bone_colors[j])
        ax.plot([gp[p, 0], gp[j, 0]], [gp[p, 1], gp[j, 1]], [gp[p, 2], gp[j, 2]],
                color=c, linewidth=lw, alpha=alpha)


def build_body_muscle_map(tracker):
    muscles = tracker.ctl.muscle_char.muscles
    body_to_muscles = {i: [] for i in range(len(BODY_NAMES))}
    for mi, mus in enumerate(muscles):
        for wp in mus.waypoints:
            for bname in wp.bodies:
                if bname in BODY_NAMES:
                    body_to_muscles[BODY_NAMES.index(bname)].append(mi)
    return [list(set(v)) for v in body_to_muscles.values()]


def activation_to_colors(a_t, body_to_muscles):
    body_act = np.zeros(len(BODY_NAMES))
    for b, idxs in enumerate(body_to_muscles):
        if idxs:
            body_act[b] = np.mean(a_t[idxs])
    body_act = np.clip(body_act, 0.0, 1.0)
    return np.stack([0.7 + 0.3 * body_act, 0.7 * (1 - body_act), 0.7 * (1 - body_act)], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="轨迹 npz (walk_muscle_demo_mujoco.py 输出)")
    parser.add_argument("--gif", default="output/mj_walk.gif")
    parser.add_argument("--frames", default="output/mj_walk_frames")
    parser.add_argument("--video", action="store_true", help="同时输出 mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--color-act", action="store_true", help="按肌肉激活给骨骼上色")
    parser.add_argument("--nframes", type=int, default=60)
    parser.add_argument("--dur", type=int, default=100)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    d = np.load(args.npz)
    actual = d["actual"]          # (T,23,3)
    ref = d.get("ref", np.zeros((0, 23, 3)))
    a = d.get("a", None)

    body_to_muscles = None
    if args.color_act:
        tracker = DirectMuscleTrackerMujoco(use_scale_map=True)
        body_to_muscles = build_body_muscle_map(tracker)

    T = actual.shape[0]
    has_ref = ref.shape[0] >= T
    print(f"实际轨迹: {actual.shape}  参考: {'有' if has_ref else '无'}  color_act={args.color_act}")

    os.makedirs(args.frames, exist_ok=True)
    os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)

    n = min(T, args.nframes)
    idxs = np.linspace(0, T - 1, n).astype(int)

    pelvis_z = actual[:, 0, 2]
    normal = pelvis_z > 0.5
    vis = actual[normal] if normal.any() else actual
    c = (vis.max(axis=(0, 1)) + vis.min(axis=(0, 1))) / 2.0
    r = float((vis.max(axis=(0, 1)) - vis.min(axis=(0, 1))).max()) * 0.6 + 0.5

    pil_frames = []
    for k, i in enumerate(idxs):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        if args.color_act and a is not None and body_to_muscles is not None:
            colors = activation_to_colors(a[i], body_to_muscles)
            draw_skeleton(ax, actual[i], BIO_PARENTS, None, lw=2.6, per_bone_colors=colors)
        else:
            draw_skeleton(ax, actual[i], BIO_PARENTS, "tab:blue", lw=2.2)
        if has_ref:
            draw_skeleton(ax, ref[i], BIO_PARENTS, "tab:red", lw=1.4, alpha=0.6)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        title = f"step {int(i)}  blue=muscle red=ref"
        if args.color_act:
            title += " (bone=activation)"
        ax.set_title(title)
        ax.view_init(elev=20, azim=-60)
        fpath = os.path.join(args.frames, f"frame_{k:03d}.png")
        fig.savefig(fpath, dpi=110)
        plt.close(fig)
        pil_frames.append(Image.open(fpath).convert("RGB"))

    if pil_frames:
        pil_frames[0].save(args.gif, save_all=True, append_images=pil_frames[1:],
                           duration=args.dur, loop=0)
        print(f"GIF 已保存: {args.gif} ({len(pil_frames)} 帧)")
        print(f"关键帧: {args.frames}/")

    if args.video and pil_frames:
        import imageio
        video_path = os.path.splitext(args.gif)[0] + ".mp4"
        imageio.mimsave(video_path, [np.array(f) for f in pil_frames], fps=args.fps)
        print(f"视频已保存: {video_path} ({len(pil_frames)} 帧 @ {args.fps}fps)")


if __name__ == "__main__":
    main()
