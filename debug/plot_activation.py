#!/usr/bin/env python3
"""动态显示 284 muscle activation (热图 + 逐帧动画, 纯 matplotlib, 无需 Isaac Lab / GPU)。

输入: walk_muscle_demo_mujoco.py --save-npz 保存的 npz (含 a: (T,284), 可选 muscle_names)。
输出:
  1) 整段激活热图 PNG: 横轴 = 帧, 纵轴 = 284 肌肉 (按解剖区域分组, 组间横线)
  2) 动图 GIF: 左侧整段热图 + 当前帧竖线, 右侧当前帧激活条形图 (按区域上色)

用法:
    python debug/walk_muscle_demo_mujoco.py --motion data/cmu_bio_npy/009/09_12.npy \
        --save-npz output/mj_traj.npz
    python debug/plot_activation.py --npz output/mj_traj.npz \
        --png output/activation_heatmap.png --gif output/activation_anim.gif
"""

import argparse
import os

import numpy as np

# 分组判定: 按子串依次匹配, 先命中者胜 (覆盖 284 条 L/R 成对肌肉)
GROUP_RULES = [
    ("Foot/Toe", ["Flexor_Digiti_Minimi_Brevis_Foot", "Flexor_Hallucis", "Extensor_Hallucis",
                  "Flexor_Digitorum_Longus", "Extensor_Digitorum_Longus"]),
    ("Ankle", ["Tibialis", "Peroneus", "Gastrocnemius", "Soleus", "Plantaris", "Popliteus"]),
    ("Knee", ["Vastus", "Rectus_Femoris", "Bicep_Femoris", "Semitendinosus", "Semimembranosus"]),
    ("Hip", ["Gluteus", "Adductor", "Psoas", "iliacus", "Sartorius", "Gracilis",
             "Tensor_Fascia", "Piriformis", "Gemellus", "Obturator", "Quadratus_Femoris",
             "Pectineus"]),
    ("Trunk", ["Longissimus", "Multifidus", "Quadratus_Lumborum", "iliocostalis",
               "Abdominis", "Serratus_Posterior"]),
    ("Shoulder", ["Deltoid", "Pectoralis", "Latissimus", "Rhomboid", "Serratus_Anterior",
                  "Trapezius", "Infraspinatus", "Supraspinatus", "Subscapularis", "Teres",
                  "Subclavian"]),
    ("Arm", ["Bicep_Brachii", "Triceps", "Brachialis", "Brachioradialis",
             "Coracobrachialis", "Anconeous"]),
    ("Forearm/Hand", ["Carpi", "Palmaris", "Pollicis", "Extensor_Digiti_Minimi",
                      "Extensor_Digitorum1", "Flexor_Digitorum_Profundus2"]),
    ("Neck", ["Sternocleidomastoid", "Splenius", "Capitis", "Scalene", "Platysma", "Omohyoid"]),
]
GROUP_COLORS = {
    "Foot/Toe": "#d62728", "Ankle": "#ff7f0e", "Knee": "#2ca02c", "Hip": "#1f77b4",
    "Trunk": "#9467bd", "Shoulder": "#8c564b", "Arm": "#e377c2",
    "Forearm/Hand": "#7f7f7f", "Neck": "#17becf",
}


def group_of(name: str) -> str:
    for g, keys in GROUP_RULES:
        for k in keys:
            if k in name:
                return g
    return "Other"


def sort_muscles(names):
    """按 (区域, 左右) 排序, 返回 (sorted_idx, group_boundaries, group_labels)。"""
    groups = [group_of(n) for n in names]
    order = sorted(range(len(names)), key=lambda i: (groups[i], names[i].split("_")[0] != "L", names[i]))
    boundaries, labels = [], []
    prev = None
    for pos, i in enumerate(order):
        if groups[i] != prev:
            boundaries.append(pos)
            labels.append(groups[i])
            prev = groups[i]
    boundaries.append(len(names))
    return np.asarray(order), boundaries, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="轨迹 npz (含 a: (T,284))")
    parser.add_argument("--png", default="output/activation_heatmap.png", help="整段热图输出路径")
    parser.add_argument("--gif", default=None, help="激活动画 GIF 输出路径 (可选)")
    parser.add_argument("--max-frames", type=int, default=120, help="动画最多帧数")
    parser.add_argument("--fps", type=int, default=10, help="GIF 帧率")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    d = np.load(args.npz, allow_pickle=True)
    a = d["a"]  # (T, 284)
    if "muscle_names" in d:
        names = [str(n) for n in d["muscle_names"]]
    else:
        names = [f"muscle_{i}" for i in range(a.shape[1])]

    T, M = a.shape
    order, bounds, labels = sort_muscles(names)
    a_sorted = a[:, order]  # (T, M)

    os.makedirs(os.path.dirname(args.png) or ".", exist_ok=True)

    # ---------- 1) 整段热图 ----------
    fig, ax = plt.subplots(figsize=(12, 14))
    im = ax.imshow(a_sorted.T, aspect="auto", origin="upper", cmap="viridis",
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="white", linewidth=1.2)
    ticks = [(bounds[i] + bounds[i + 1]) / 2 for i in range(len(labels))]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("frame")
    ax.set_ylabel("284 muscles (grouped)")
    ax.set_title(f"Muscle activation over time ({M} muscles, {T} frames)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="activation")
    fig.savefig(args.png, dpi=120)
    plt.close(fig)
    print(f"热图已保存: {args.png} (分组: {labels})")

    # ---------- 2) 激活动画 ----------
    if not args.gif:
        return
    os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
    n = min(T, args.max_frames)
    idxs = np.linspace(0, T - 1, n).astype(int)

    frames = []
    for f in idxs:
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.15)

        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(a_sorted.T, aspect="auto", origin="upper", cmap="viridis",
                   vmin=0.0, vmax=1.0, interpolation="nearest")
        for b in bounds[1:-1]:
            ax1.axhline(b - 0.5, color="white", linewidth=0.8)
        ax1.axvline(f, color="red", linewidth=2.0)
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel("frame")
        ax1.set_title(f"activation history (frame {f}/{T})")

        ax2 = fig.add_subplot(gs[1])
        ypos = np.arange(M)[::-1]
        col = [GROUP_COLORS.get(group_of(names[i]), "gray") for i in order]
        ax2.barh(ypos, a_sorted[f], color=col, height=0.9)
        ax2.axvline(0.5, color="gray", linewidth=0.6, linestyle="--")
        for b in bounds[1:-1]:
            ax2.axhline(M - b - 0.5, color="black", linewidth=0.8)
        ax2.set_ylim(-1, M)
        ax2.set_xlim(0.0, 1.0)
        ax2.set_yticks([M - (bounds[i] + bounds[i + 1]) / 2 for i in range(len(labels))])
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel("activation")
        ax2.set_title(f"current activations (frame {f})")
        fig.savefig("/tmp/_act_frame.png", dpi=100)
        plt.close(fig)
        frames.append(Image.open("/tmp/_act_frame.png").convert("RGB"))

    frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0)
    os.remove("/tmp/_act_frame.png")
    print(f"动画已保存: {args.gif} ({n} 帧 @ {args.fps}fps)")


if __name__ == "__main__":
    main()
