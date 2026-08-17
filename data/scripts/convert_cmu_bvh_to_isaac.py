#!/usr/bin/env python3
"""把 CMU MoCap 数据集的 BVH 文件转换为 BIO 骨骼的 poselib SkeletonMotion (.npy)。

与 convert_lafan_bvh_to_isaac.py 逻辑一致 (T-pose 归一化 -> 局部旋转复制 retarget ->
踝关节修正 -> Y-up->Z-up -> 高度对齐 -> fps 重采样), 针对 CMU 数据的差异:
  1. CMU 每个 BVH 的「第一帧」是 Bruce Hahne 合成的统一 T-pose, 需剔除 (--keep-tpose 可保留);
  2. T-pose 参考直接用第一帧的全局姿态 (对任意 subject 都鲁棒);
  3. 骨架映射表使用 CMU 的 MotionBuilder 命名 (含 LeftToeBase -> BIO 两个脚趾)。

用法 (仓库根目录执行):
    python data/scripts/convert_cmu_bvh_to_isaac.py \
        --input data/cmu_mocap --output data/cmu_bio_npy \
        --subjects 08 09 35 91 104 105

或批量转换所有已下载 subject:
    python data/scripts/convert_cmu_bvh_to_isaac.py \
        --input data/cmu_mocap --output data/cmu_bio_npy

输出: <output>/<subject>/<subject>_<idx>.npy (BIO 骨架, 30fps, 单位: 米)
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

# 加入仓库根路径
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from data.scripts.lafan_utils import read_bvh  # noqa: E402
from poselib.skeleton.skeleton3d import (  # noqa: E402
    SkeletonMotion,
    SkeletonState,
    SkeletonTree,
)

DEFAULT_CONFIG = os.path.join(current_dir, "..", "configs", "bvh_to_bio.yaml")
BIO_XML = os.path.join(repo_root, "protomotions", "data", "assets", "mjcf", "bio.xml")

# ---------------------------------------------------------------------------
# 重定向配置 (关节映射 + source_length_scale 见 data/configs/bvh_to_bio.yaml)
# ---------------------------------------------------------------------------
def load_config(config_path: str, scale_override=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if scale_override is not None:
        cfg["source_length_scale"] = scale_override
    return cfg


# ---------------------------------------------------------------------------
# 行走动作筛选 (基于 CMU 官方动作索引 cmu-mocap-index-text.txt)
# ---------------------------------------------------------------------------
WALK_RE = re.compile(r"walk", re.IGNORECASE)
EXCLUDE_RE = re.compile(
    r"\b(run|jog|jump|leap|dance|moonwalk|kick|punch|throw|catch|basketball|"
    r"soccer|football|baseball|golf|swim|tennis|climb|swing|motorcycle|yoga|"
    r"squat|lunge|cartwheel|flip|handstand|stool|chair|bench|ladder|playground|dribble)\b",
    re.IGNORECASE,
)


def parse_cmu_index(index_path: str):
    """解析 CMU 索引文本, 返回 {<编号>: <动作描述>}, 如 {"08_01": "walk"}。"""
    motion_descs = {}
    if not os.path.exists(index_path):
        print(f"  [warn] 索引文件不存在: {index_path}")
        return motion_descs
    with open(index_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Subject") or line.startswith("CMU") or line.startswith("Carnegie"):
                continue
            m = re.match(r"^(\d+_\d+)\s+(.+)$", line)
            if m:
                motion_descs[m.group(1)] = m.group(2)
    return motion_descs


def is_walk_motion(desc: str) -> bool:
    """判断某动作描述是否为「行走」 (含 walk/stride/step, 且排除跑步/跳跃/舞蹈/球类等)。"""
    desc = desc or ""
    return bool(WALK_RE.search(desc)) and not EXCLUDE_RE.search(desc)


# ---------------------------------------------------------------------------
# 转换逻辑
# ---------------------------------------------------------------------------
def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _rotate_global(m: SkeletonMotion, r: R) -> SkeletonMotion:
    """把 motion 的全局旋转与根位移绕世界系旋转 (y-up -> z-up 用绕 x +90 度)。"""
    gr = _as_numpy(m.global_rotation).astype(np.float64)
    rt = _as_numpy(m.root_translation).astype(np.float64)
    gr_new = (r * R.from_quat(gr.reshape(-1, 4))).as_quat().reshape(gr.shape)
    rt_new = r.apply(rt)
    st = SkeletonState.from_rotation_and_root_translation(
        m.skeleton_tree,
        torch.from_numpy(gr_new).float(),
        torch.from_numpy(rt_new).float(),
        is_local=False,
    )
    return SkeletonMotion.from_skeleton_state(st, fps=float(m.fps))


def _foot_on_ground(m: SkeletonMotion) -> SkeletonMotion:
    """整体平移使最低关节 (脚底) z = 0, 保证脚贴地。"""
    min_z = float(_as_numpy(m.global_translation)[..., 2].min())
    if abs(min_z) > 1e-4:
        rt = _as_numpy(m.root_translation)
        rt = rt + np.array([0.0, 0.0, -min_z])
        st = SkeletonState.from_rotation_and_root_translation(
            m.skeleton_tree,
            m.global_rotation,
            torch.from_numpy(rt).float(),
            is_local=False,
        )
        return SkeletonMotion.from_skeleton_state(st, fps=float(m.fps))
    return m


def _slice_motion(m: SkeletonMotion, start: int, end: int = None) -> SkeletonMotion:
    """按帧切片 (保留骨架树与 fps)。"""
    return SkeletonMotion(m.tensor[start:end], m.skeleton_tree, m.is_local, fps=m.fps)


def convert_cmu_to_bio_npy(input_path: str, output_path: str, cfg: dict, keep_tpose: bool = False):
    """转换单个 CMU BVH 文件为 BIO 骨架的 .npy (poselib retarget_to 规范管线)。"""
    anim = read_bvh(input_path)

    # 1. 源 SkeletonMotion (y-up, cm)
    src_tree = SkeletonTree(
        anim.bones,
        torch.tensor(anim.parents, dtype=torch.long),
        torch.from_numpy(anim.offsets).float(),
    )
    src_local_rot = np.roll(anim.quats, -1, axis=-1)  # wxyz -> xyzw
    src_state = SkeletonState.from_rotation_and_root_translation(
        src_tree,
        torch.from_numpy(src_local_rot).float(),
        torch.from_numpy(anim.pos[:, 0]).float(),  # cm
        is_local=True,
    )
    src_motion = SkeletonMotion.from_skeleton_state(src_state, fps=float(anim.fps))

    # 2. source_tpose = 第一帧 (Bruce 合成的统一 T-pose)
    source_tpose = SkeletonState(src_motion.tensor[0], src_motion.skeleton_tree, src_motion.is_local)

    # 3. target_tpose = BIO zero_pose
    target_tree = SkeletonTree.from_mjcf(BIO_XML)
    target_tpose = SkeletonState.zero_pose(target_tree)

    # 4. poselib retarget (源 -> 目标, source_length_scale 显式)
    rotation = torch.tensor(cfg["rotation_to_target_skeleton"], dtype=torch.float32)
    new_motion = src_motion.retarget_to(
        joint_mapping=cfg["joint_mapping"],
        source_tpose_local_rotation=source_tpose.local_rotation,
        source_tpose_root_translation=source_tpose.root_translation,
        target_skeleton_tree=target_tree,
        target_tpose_local_rotation=target_tpose.local_rotation,
        target_tpose_root_translation=target_tpose.root_translation,
        rotation_to_target_skeleton=rotation,
        scale_to_target_skeleton=cfg["source_length_scale"],
    )

    # 5. 剔除首帧合成 T-pose
    if not keep_tpose:
        if new_motion.tensor.shape[0] <= 1:
            print(f"  [warn] {input_path}: 只有 T-pose 一帧, 跳过")
            return
        new_motion = _slice_motion(new_motion, 1)

    # 6. y-up -> z-up
    new_motion = _rotate_global(new_motion, R.from_euler("x", 90, degrees=True))

    # 7. 脚贴地
    new_motion = _foot_on_ground(new_motion)

    # 8. FPS 重采样: 120 -> 30
    fps_src = round(anim.fps)
    target_fps = 30
    if fps_src != target_fps:
        T_src = int(new_motion.global_rotation.shape[0])
        if fps_src % target_fps == 0:
            skip = int(fps_src / target_fps)
            sliced = SkeletonState.from_rotation_and_root_translation(
                new_motion.skeleton_tree,
                new_motion.global_rotation[::skip],
                new_motion.root_translation[::skip],
                is_local=False,
            )
        else:
            indices = np.round(
                np.linspace(0, T_src - 1, int(T_src * target_fps / fps_src))
            ).astype(int)
            sliced = SkeletonState.from_rotation_and_root_translation(
                new_motion.skeleton_tree,
                new_motion.global_rotation[indices],
                new_motion.root_translation[indices],
                is_local=False,
            )
        new_motion = SkeletonMotion.from_skeleton_state(sliced, fps=target_fps)

    # 9. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_motion.to_file(output_path)


def main():
    parser = argparse.ArgumentParser(description="CMU BVH -> BIO SkeletonMotion (.npy)")
    parser.add_argument("--input", type=str, required=True, help="CMU bvh 目录 (含 subject 子目录)")
    parser.add_argument("--output", type=str, required=True, help="输出目录 (.npy)")
    parser.add_argument(
        "--subjects", nargs="+", type=str, default=None,
        help="只转换指定 subject, 如 08 09 35 (默认转换全部)",
    )
    parser.add_argument(
        "--keep-tpose", action="store_true",
        help="保留每段第一帧合成 T-pose (默认剔除)",
    )
    parser.add_argument(
        "--only-walk", action="store_true",
        help="只转换索引标记为「行走」的动作 (基于 CMU 官方动作索引筛选)",
    )
    parser.add_argument(
        "--index", type=str, default=None,
        help="CMU 动作索引文件路径 (默认: data/cmu_mocap/_manifest/cmu-mocap-index-text.txt)",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="重定向配置文件 (默认 data/configs/bvh_to_bio.yaml)",
    )
    parser.add_argument(
        "--scale", type=float, default=None,
        help="覆盖配置文件里的 source_length_scale",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.scale)
    print(f"重定向配置: source_length_scale={cfg['source_length_scale']}, "
          f"mapping={len(cfg['joint_mapping'])} 关节")

    # 收集 BVH 文件
    if args.subjects:
        files = []
        for s in args.subjects:
            files.extend(sorted(glob.glob(os.path.join(args.input, s.zfill(3), "*.bvh"))))
    else:
        files = sorted(glob.glob(os.path.join(args.input, "**", "*.bvh"), recursive=True))

    # 行走筛选
    if args.only_walk:
        index_path = args.index or os.path.join(
            args.input, "_manifest", "cmu-mocap-index-text.txt"
        )
        walk_descs = parse_cmu_index(index_path)
        n_before = len(files)
        kept = []
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]  # 如 08_01
            desc = walk_descs.get(base, "")
            if is_walk_motion(desc):
                kept.append(f)
            else:
                print(f"  [跳过] {base}: {desc or '(无描述)'}")
        files = kept
        print(f"行走筛选: {n_before} -> {len(files)} 个文件")

    print(f"找到 {len(files)} 个 BVH 文件, 开始转换...")
    ok = fail = 0
    for i, f in enumerate(files, 1):
        rel = os.path.relpath(f, args.input)
        out = os.path.join(args.output, rel[:-4] + ".npy")
        try:
            convert_cmu_to_bio_npy(f, out, cfg, keep_tpose=args.keep_tpose)
            ok += 1
        except Exception as e:
            print(f"  [fail] {rel}: {e}")
            fail += 1
        if i % 20 == 0 or i == len(files):
            print(f"  进度 {i}/{len(files)}  ok={ok} fail={fail}")

    print(f"\n转换完成: 成功 {ok}, 失败 {fail}")
    if fail:
        print("部分文件失败, 详见上方日志。")


if __name__ == "__main__":
    main()
