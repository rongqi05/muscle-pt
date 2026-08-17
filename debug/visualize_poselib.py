#!/usr/bin/env python3
"""用 poselib matplotlib 3D 可视化重定向后的 BIO 运动 (替代 Isaac Sim 渲染)。

背景: Isaac Sim 5.1.0 的 Hydra/RTX 渲染器不识别 RTX 5060 (Blackwell 架构),
GUI 与离屏渲染都会在 createHydraEngine 处段错误。本脚本用纯 CPU 的
matplotlib 3D 骨架图, 无需 GPU 渲染, 同样能旋转视角、逐帧检查穿透/扭斜。

用法:
    # 交互式查看 (键盘控制, 见下方按键说明)
    python debug/visualize_poselib.py --motion data/cmu_bio_npy/008/08_01.npy

    # 离屏保存关键帧 PNG (无窗口环境也能出图)
    python debug/visualize_poselib.py --motion data/cmu_bio_npy/008/08_01.npy --save output/frames

交互按键 (plot_skeleton_motion_interactive):
    x  播放/暂停        z/c  上一帧/下一帧
    a/d 前/后 20 帧      w  循环播放
    v/b 加速/减速        n  退出
    鼠标拖动旋转 3D 视角, 滚轮缩放
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from poselib.skeleton.skeleton3d import SkeletonMotion  # noqa: E402


def get_global_and_parents(motion: SkeletonMotion):
    gp = motion.global_translation.numpy()  # (T, J, 3)
    parents = motion.skeleton_tree.parent_indices.numpy()
    return gp, parents


def draw_skeleton(ax, gp_frame, parents, color="tab:blue", lw=2.0):
    """绘制单帧骨架连线 (关节 -> 父关节)。"""
    for j in range(1, len(parents)):
        p = int(parents[j])
        if p < 0:
            continue
        ax.plot(
            [gp_frame[p, 0], gp_frame[j, 0]],
            [gp_frame[p, 1], gp_frame[j, 1]],
            [gp_frame[p, 2], gp_frame[j, 2]],
            color=color, linewidth=lw,
        )


def set_equal_lim(ax, gp):
    c = (gp.max(axis=(0, 1)) + gp.min(axis=(0, 1))) / 2.0
    r = float((gp.max(axis=(0, 1)) - gp.min(axis=(0, 1))).max()) * 0.65
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def save_frames(motion: SkeletonMotion, out_dir: str, n: int = 8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gp, parents = get_global_and_parents(motion)
    os.makedirs(out_dir, exist_ok=True)
    indices = np.linspace(0, len(motion) - 1, n).astype(int)
    for idx in indices:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        draw_skeleton(ax, gp[idx], parents)
        set_equal_lim(ax, gp)
        ax.set_title(f"frame {idx}/{len(motion)}")
        out = os.path.join(out_dir, f"frame_{idx:04d}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  保存 {out}")
    print(f"关键帧已保存到 {out_dir}/")


def save_gif(motion: SkeletonMotion, out_path: str, fps: int = 10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    gp, parents = get_global_and_parents(motion)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    set_equal_lim(ax, gp)

    def update(i):
        ax.clear()
        draw_skeleton(ax, gp[i], parents)
        set_equal_lim(ax, gp)
        ax.set_title(f"frame {i}/{len(motion)}")

    ani = FuncAnimation(fig, update, frames=len(motion), interval=1000 / fps)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ani.save(out_path, writer=PillowWriter(fps=fps))
    print(f"动画已保存到 {out_path}")
    plt.close(fig)


def interactive(motion: SkeletonMotion):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    gp, parents = get_global_and_parents(motion)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    set_equal_lim(ax, gp)
    idx = [0]

    def draw():
        ax.clear()
        draw_skeleton(ax, gp[idx[0]], parents)
        set_equal_lim(ax, gp)
        ax.set_title(f"frame {idx[0]}/{len(motion)}   (拖动旋转视角, 关闭窗口退出)")

    def on_key(event):
        if event.key == "right":
            idx[0] = (idx[0] + 1) % len(motion)
        elif event.key == "left":
            idx[0] = (idx[0] - 1) % len(motion)
        elif event.key == "up":
            idx[0] = (idx[0] + 20) % len(motion)
        elif event.key == "down":
            idx[0] = (idx[0] - 20) % len(motion)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        idx[0] = (idx[0] + 1) % len(motion)
        draw()

    FuncAnimation(fig, update, interval=80, blit=False)
    draw()
    print("交互式播放: 方向键 ←→ 逐帧, ↑↓ 快进, 鼠标拖动旋转, 关闭窗口退出")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="matplotlib 3D 骨架可视化")
    parser.add_argument("--motion", type=str, required=True, help=".npy 文件或通配")
    parser.add_argument("--save", type=str, default=None, help="离屏保存关键帧 PNG 到目录")
    parser.add_argument("--gif", type=str, default=None, help="保存动画 GIF 到路径")
    args = parser.parse_args()

    files = sorted(glob.glob(args.motion))
    assert files, f"未找到: {args.motion}"
    motion = SkeletonMotion.from_file(files[0])
    names = list(motion.skeleton_tree.node_names)
    print(f"motion: {files[0]}")
    print(f"  帧数={len(motion)}  fps={motion.fps}  关节={len(names)}")

    if args.save:
        save_frames(motion, args.save)
    elif args.gif:
        save_gif(motion, args.gif)
    else:
        print("观察要点: 1)手臂是否穿透躯干 2)躯干是否扭斜 3)腿/脚摆动是否自然")
        interactive(motion)


if __name__ == "__main__":
    main()
