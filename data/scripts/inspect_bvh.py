#!/usr/bin/env python3
"""检查 BVH 文件的结构与质量。

用法:
    python data/scripts/inspect_bvh.py path/to/walk.bvh
    python data/scripts/inspect_bvh.py path/to/walk.bvh --scale 0.01

打印内容:
  - 文件路径 / 总帧数 / 帧时间 / FPS / 根关节 / 关节总数
  - 完整层级 (parent 关系)
  - 每个关节的 OFFSET 与 CHANNELS 声明 (含 Euler 旋转顺序)
  - 根位移范围 (水平面 + 高度)
  - 估计骨架高度 / 髋到脚距离
  - 估计根位移 (直线 + 路径) 与行走速度
  - 问题检测: 缺失腿部关节 / 重复名字 / 异常通道顺序 / NaN·Inf / 零长度骨骼 / 可疑缩放

本脚本只读取 BVH, 不修改文件。
"""
import argparse
import os
import re
import sys

import numpy as np

# BVH CHANNELS 名称 -> 单轴缩写
CHANNEL_MAP = {
    "Xposition": "xp", "Yposition": "yp", "Zposition": "zp",
    "Xrotation": "xr", "Yrotation": "yr", "Zrotation": "zr",
}

# 常见腿部关节名 (用于缺失检测)
EXPECTED_LEG_JOINTS = [
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "lhipjoint", "lknee", "lankle", "ltoes",
    "rhipjoint", "rknee", "rankle", "rtoes",
    "LFemur", "LTibia", "LFoot", "RFemur", "RTibia", "RFoot",
    "lfemur", "ltibia", "lfoot", "rfemur", "rtibia", "rfoot",
]


class BVHNode:
    def __init__(self, name, node_type, parent):
        self.name = name
        self.node_type = node_type  # "root" | "joint" | "end"
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels = []  # 如 ["Xposition","Yposition","Zposition","Zrotation","Yrotation","Xrotation"]
        self.euler_order = None  # 如 "ZYX"
        self.children = []

    def euler_order_str(self):
        return self.euler_order if self.euler_order else "-"


def parse_structure(path):
    """解析 BVH HIERARCHY 段, 返回 (nodes, root_idx)。"""
    nodes = []
    stack = []
    root_idx = None
    in_end_site = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        s = line.strip()
        if not s or s.startswith("MOTION"):
            break
        if s.startswith("HIERARCHY") or s == "{" or s.startswith("}"):
            # "}" 结束当前节点
            if s.startswith("}") and stack:
                stack.pop()
            continue

        m = re.match(r"ROOT\s+(\S+)", s, re.IGNORECASE)
        if m:
            node = BVHNode(m.group(1), "root", -1)
            nodes.append(node)
            root_idx = len(nodes) - 1
            stack.append(root_idx)
            in_end_site = False
            continue

        m = re.match(r"JOINT\s+(\S+)", s, re.IGNORECASE)
        if m:
            parent = stack[-1] if stack else -1
            node = BVHNode(m.group(1), "joint", parent)
            nodes.append(node)
            if parent >= 0:
                nodes[parent].children.append(len(nodes) - 1)
            stack.append(len(nodes) - 1)
            in_end_site = False
            continue

        if "End Site" in s:
            in_end_site = True
            continue

        m = re.match(r"OFFSET\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)", s)
        if m and stack:
            nodes[stack[-1]].offset = np.array([float(m.group(i)) for i in (1, 2, 3)])
            continue

        m = re.match(r"CHANNELS\s+(\d+)\s+(.+)", s)
        if m and stack:
            n = int(m.group(1))
            chans = m.group(2).split()
            nodes[stack[-1]].channels = chans[:n]
            rot_order = "".join(
                c[0].upper() for c in chans if "rotation" in c.lower()
            )
            nodes[stack[-1]].euler_order = rot_order if rot_order else None
            continue

    return nodes, root_idx


def parse_motion(path, root_channels):
    """解析 MOTION 段, 返回 (frames, frametime, root_pos)。"""
    frames = 0
    frametime = 0.033333
    data_lines = []
    in_motion = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("MOTION"):
                in_motion = True
                continue
            if not in_motion:
                continue
            m = re.match(r"Frames:\s*(\d+)", s, re.IGNORECASE)
            if m:
                frames = int(m.group(1))
                continue
            m = re.match(r"Frame Time:\s*([\d.eE+-]+)", s, re.IGNORECASE)
            if m:
                frametime = float(m.group(1))
                continue
            if s and s[0].replace(".", "").replace("-", "").replace("+", "").replace("e", "").replace("E", "").isdigit():
                data_lines.append(s)

    # 根通道: 前 3 个是位置 (Xposition Yposition Zposition)
    root_pos = np.zeros((frames, 3))
    pos_cols = [i for i, c in enumerate(root_channels) if "position" in c.lower()]
    if len(pos_cols) >= 3:
        for fi, s in enumerate(data_lines[:frames]):
            vals = s.split()
            try:
                root_pos[fi] = [float(vals[i]) for i in pos_cols[:3]]
            except (IndexError, ValueError):
                pass
    return frames, frametime, root_pos


def print_hierarchy(nodes, root_idx):
    def rec(idx, depth):
        node = nodes[idx]
        indent = "  " * depth
        chan = " ".join(node.channels) if node.channels else "-"
        euler = f"euler={node.euler_order_str()}" if node.euler_order else ""
        print(f"{indent}{node.name} [{node.node_type}] parent={nodes[node.parent].name if node.parent >= 0 else 'None'} "
              f"offset=({node.offset[0]:.3f},{node.offset[1]:.3f},{node.offset[2]:.3f}) "
              f"channels=[{chan}] {euler}")
        for c in node.children:
            rec(c, depth + 1)
    rec(root_idx, 0)


def main():
    parser = argparse.ArgumentParser(description="检查 BVH 文件结构")
    parser.add_argument("path", help="BVH 文件路径")
    parser.add_argument("--scale", type=float, default=None,
                        help="单位换算因子 (如 cm->m 用 0.01); 默认自动估计")
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        sys.exit(1)

    nodes, root_idx = parse_structure(path)
    if root_idx is None:
        print("无法解析 BVH HIERARCHY")
        sys.exit(1)

    root = nodes[root_idx]
    frames, frametime, root_pos = parse_motion(path, root.channels)
    fps = 1.0 / frametime if frametime > 0 else 0.0

    print("=" * 70)
    print("BVH 文件检查报告")
    print("=" * 70)
    print(f"文件路径: {path}")
    print(f"总帧数: {frames}")
    print(f"帧时间: {frametime}")
    print(f"FPS: {fps:.2f}")
    print(f"根关节: {root.name} (channels={root.channels})")
    print(f"关节总数: {len(nodes)} (含 root 与 End Site)")

    # 完整层级
    print("\n--- 完整层级 ---")
    print_hierarchy(nodes, root_idx)

    # 通道顺序统计
    print("\n--- 通道/Euler 顺序统计 ---")
    from collections import Counter
    orders = Counter(n.euler_order for n in nodes if n.euler_order)
    for o, c in orders.items():
        print(f"  Euler 顺序 {o}: {c} 个关节")

    # 骨架几何
    offsets = np.array([n.offset for n in nodes])
    bone_len = np.linalg.norm(offsets, axis=1)
    # 估计骨架高度: 根到最高点 + 根到最低点 (用层级 FK 近似)
    # 这里用 offset 的累计 (父->子) 近似, 更精确用 FK, 但 offset 累计已足够估计
    def global_offset(idx):
        if nodes[idx].parent < 0:
            return nodes[idx].offset
        return nodes[idx].offset + global_offset(nodes[idx].parent)
    glob = np.array([global_offset(i) for i in range(len(nodes))])
    height_axis = np.argmax(np.abs(glob).max(axis=0))  # 最大变化轴近似高度轴
    height_range = glob[:, height_axis].max() - glob[:, height_axis].min()
    print(f"\n--- 骨架几何 (单位: BVH 原始单位) ---")
    print(f"估计骨架高度: {height_range:.2f}")
    print(f"最大骨段长度: {bone_len.max():.2f}")

    # 髋到脚距离: 找髋(UpLeg 的父)到 Toe/Foot
    hip_names = ["Hips", "hips", "pelvis", "Pelvis"]
    foot_names = ["LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase", "lfoot", "rfoot"]
    name2idx = {n.name: i for i, n in enumerate(nodes)}
    hip_idx = next((name2idx[nm] for nm in hip_names if nm in name2idx), None)
    foot_idx = next((name2idx[nm] for nm in foot_names if nm in name2idx), None)
    if hip_idx is not None and foot_idx is not None:
        hip_foot = np.linalg.norm(glob[hip_idx] - glob[foot_idx])
        print(f"髋到脚距离 ({nodes[hip_idx].name} -> {nodes[foot_idx].name}): {hip_foot:.2f}")

    # 根位移范围
    if root_pos.shape[0] > 0:
        rp = root_pos
        print("\n--- 根位移 ---")
        print(f"根位置范围: x=({rp[:,0].min():.2f},{rp[:,0].max():.2f}) "
              f"y=({rp[:,1].min():.2f},{rp[:,1].max():.2f}) "
              f"z=({rp[:,2].min():.2f},{rp[:,2].max():.2f})")
        # 高度轴: 用变化最大的轴 vs 前进轴
        dur = frames / fps if fps > 0 else 0
        disp_line = np.linalg.norm(rp[-1] - rp[0])
        path_len = np.linalg.norm(np.diff(rp, axis=0), axis=1).sum()
        print(f"直线位移: {disp_line:.2f} (路径 {path_len:.2f})  时长 {dur:.1f}s")
        print(f"平均速度: {disp_line / dur:.2f} 单位/s  路径速度: {path_len / dur:.2f} 单位/s")

    # 单位估计
    print("\n--- 单位估计 ---")
    if height_range > 50:
        print(f"骨架高度 {height_range:.1f} > 50, 推测单位为 cm -> 米换算因子 0.01")
        auto_scale = 0.01
    elif height_range > 5:
        print(f"骨架高度 {height_range:.1f} 在 5~50, 推测单位为 cm (偏小) 或 dm")
        auto_scale = 0.01
    elif height_range > 1:
        print(f"骨架高度 {height_range:.2f} 在 1~5, 推测单位为 m (但骨架可能偏小)")
        auto_scale = 1.0
    else:
        print(f"骨架高度 {height_range:.3f} < 1, 推测单位为 m 且骨架异常偏小")
        auto_scale = 1.0
    scale = args.scale if args.scale is not None else auto_scale
    print(f"使用换算因子: {scale}  (骨架高度 x scale = {height_range * scale:.2f} m)")

    # 问题检测
    print("\n--- 问题检测 ---")
    issues = []
    # 缺失腿部关节
    present = set(n.name for n in nodes)
    expected_found = [nm for nm in EXPECTED_LEG_JOINTS if nm in present]
    if len(expected_found) < 4:
        issues.append(f"腿部关节缺失或命名非常规 (只找到 {expected_found})")
    else:
        print(f"腿部关节: 找到 {expected_found}")
    # 重复名字
    dup = [nm for nm, c in Counter(present).items() if c > 1]
    if dup:
        issues.append(f"重复关节名: {dup}")
    # 异常通道顺序
    for n in nodes:
        if n.channels and n.node_type in ("root", "joint"):
            # 检查是否 rotation 通道
            if not any("rotation" in c.lower() for c in n.channels):
                issues.append(f"关节 {n.name} 无 rotation 通道: {n.channels}")
    # NaN/Inf
    if root_pos.shape[0] > 0 and (not np.isfinite(root_pos).all()):
        issues.append("根位置含 NaN/Inf")
    # 零长度骨骼
    zero_bones = [n.name for n in nodes if np.linalg.norm(n.offset) < 1e-6 and n.node_type in ("root", "joint")]
    if zero_bones:
        print(f"零长度骨骼 (可能为占位关节): {zero_bones}")
    # 可疑缩放
    if height_range < 50 and scale == 0.01:
        issues.append(f"骨架高度 {height_range:.1f} (cm) 异常偏小, 疑似数据整体缩放, 建议人工确认 scale")
    if height_range < 2:
        issues.append(f"骨架高度 {height_range:.2f} (m) 异常偏小 (<2m 人形下限), 疑似数据被缩放 ~{2/height_range:.0f}x")

    if issues:
        print("检测到以下问题:")
        for it in issues:
            print(f"  [警告] {it}")
    else:
        print("未检测到明显问题。")
    print("=" * 70)


if __name__ == "__main__":
    main()
