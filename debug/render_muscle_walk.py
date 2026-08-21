#!/usr/bin/env python3
"""渲染"肌肉驱动行走" GIF —— 纯 matplotlib, 不依赖 Isaac Sim 渲染器。

背景: 本机 RTX 5060 (Blackwell) 无法用 Isaac Sim GUI/离屏渲染, 故用离线渲染。
本脚本:
  1. (可选) 用 DirectMuscleTracker 跑一遍直接肌肉控制, 得到实际轨迹 + 参考轨迹
  2. 用 matplotlib 画 3D 骨架: 蓝 = 肌肉驱动实际, 红 = 参考 q_ref
  3. --color-act 时, 骨架骨骼按肌肉激活上色 (灰 → 红)

用法 (仓库根目录):
    python debug/render_muscle_walk.py \
        --motion data/cmu_bio_npy/009/09_12.npy --gif output/muscle_walk.gif

    # 只从已保存的 npz 重渲染 (不重新仿真):
    python debug/render_muscle_walk.py --npz output/muscle_walk_traj.npz --gif output/muscle_walk.gif
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
from protomotions.utils.direct_muscle import DirectMuscleTracker, BODY_NAMES  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# BODY_NAMES 顺序下的父节点索引 (Pelvis=0 为根, -1 表示无父)
BIO_PARENTS = [-1, 0, 1, 2, 3,
               2, 5, 6, 7,
               2, 9, 10, 11,
               0, 13, 14, 15, 15,
               0, 18, 19, 20, 20]


def draw_skeleton(ax, gp, parents, color, lw=2.0, alpha=1.0, per_bone_colors=None):
    """画骨架。per_bone_colors: 可选 (23,) RGB, 覆盖 color。"""
    for j in range(1, len(parents)):
        p = int(parents[j])
        if p < 0 or p >= len(gp):
            continue
        c = color if per_bone_colors is None else tuple(per_bone_colors[j])
        ax.plot([gp[p, 0], gp[j, 0]],
                [gp[p, 1], gp[j, 1]],
                [gp[p, 2], gp[j, 2]],
                color=c, linewidth=lw, alpha=alpha)


def build_body_muscle_map(tracker):
    """返回 body_idx -> 关联肌肉索引列表 (用于按激活给骨骼上色)。"""
    muscles = tracker.ctl.muscle_char.muscles
    body_to_muscles = {i: [] for i in range(len(BODY_NAMES))}
    for mi, mus in enumerate(muscles):
        for wp in mus.waypoints:
            for bname in wp.bodies:
                if bname in BODY_NAMES:
                    body_to_muscles[BODY_NAMES.index(bname)].append(mi)
    return [list(set(v)) for v in body_to_muscles.values()]


def activation_to_colors(a_t, body_to_muscles):
    """把一帧激活 (284,) 映射为每根骨骼的颜色 (23,3)。灰=0, 红=1。"""
    body_act = np.zeros(len(BODY_NAMES))
    for b, idxs in enumerate(body_to_muscles):
        if idxs:
            body_act[b] = np.mean(a_t[idxs])
    body_act = np.clip(body_act, 0.0, 1.0)
    # 灰 (0.7) -> 红 (1,0,0)
    return np.stack([
        0.7 + 0.3 * body_act,     # R
        0.7 * (1.0 - body_act),   # G
        0.7 * (1.0 - body_act),   # B
    ], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=str, default=None, help=".npy motion 文件 (直接仿真+渲染)")
    parser.add_argument("--npz", type=str, default=None, help="已保存的轨迹 npz (跳过仿真)")
    parser.add_argument("--gif", default="output/muscle_walk.gif", help="输出 GIF")
    parser.add_argument("--frames", default="output/muscle_walk_frames", help="关键帧目录")
    parser.add_argument("--max-frames", type=int, default=120, help="渲染最大帧数")
    parser.add_argument("--method", type=str, default="ls", choices=["ls", "lbfgs"])
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--no-scale-map", action="store_true", help="关闭肌肉 f0 补偿")
    parser.add_argument("--color-act", action="store_true", help="按肌肉激活给骨骼上色")
    parser.add_argument("--nframes", type=int, default=40, help="GIF/视频帧数 (设为 --max-frames 可渲染每一帧)")
    parser.add_argument("--dur", type=int, default=120, help="GIF 每帧毫秒")
    parser.add_argument("--video", action="store_true", help="同时输出 mp4 视频 (用 imageio-ffmpeg)")
    parser.add_argument("--fps", type=int, default=30, help="mp4 输出帧率")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    body_to_muscles = None
    if args.motion:
        tracker = DirectMuscleTracker(device="cuda:0", simulation_app=simulation_app,
                                      use_scale_map=not args.no_scale_map)
        res = tracker.track(args.motion, max_frames=args.max_frames,
                            method=args.method, max_iter=args.max_iter)
        actual = res["body_pos"]          # (T,23,3)
        a = res["a"]                      # (T,284)
        ref = tracker.kinematic_reference(args.motion, max_frames=args.max_frames)
        if args.color_act:
            body_to_muscles = build_body_muscle_map(tracker)
        # 保存轨迹供复用
        npz_out = args.npz or os.path.splitext(args.gif)[0] + "_traj.npz"
        np.savez(npz_out, actual=actual, ref=ref, a=a)
        print(f"轨迹已保存 -> {npz_out}")
        tracker.close()
    elif args.npz:
        d = np.load(args.npz)
        actual = d["actual"]
        ref = d.get("ref", np.zeros((0, 23, 3)))
        a = d.get("a", None)
        # 上色需要肌肉-骨骼映射, 但仅凭 npz 无法重建; 跳过
        if args.color_act:
            print("[warn] --npz 模式无法重建肌肉-骨骼映射, 忽略 --color-act")
    else:
        print("需提供 --motion 或 --npz")
        sys.exit(1)

    T = actual.shape[0]
    has_ref = ref.shape[0] > 0 and ref.shape[0] >= T
    print(f"实际轨迹: {actual.shape}  参考: {'有' if has_ref else '无'}  color_act={args.color_act}")

    os.makedirs(args.frames, exist_ok=True)
    os.makedirs(os.path.dirname(args.gif), exist_ok=True)

    n_frames = min(T, args.nframes)
    idxs = np.linspace(0, T - 1, n_frames).astype(int)

    # 视野基于骨盆高度正常帧
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
        title = f"step {int(i)}  蓝=肌肉驱动 红=参考"
        if args.color_act:
            title += "  (骨骼色=激活)"
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
        frames_np = [np.array(f) for f in pil_frames]
        imageio.mimsave(video_path, frames_np, fps=args.fps)
        print(f"视频已保存: {video_path} ({len(pil_frames)} 帧 @ {args.fps}fps)")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
