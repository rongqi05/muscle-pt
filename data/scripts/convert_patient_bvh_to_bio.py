#!/usr/bin/env python3
"""病人 BVH (Character1 骨架) -> BIO 骨架 poselib SkeletonMotion (.npy)。

与 CMU 管线 (convert_cmu_bvh_to_isaac.py) 同模式, 病人数据差异:
  1. Character1 骨架 (37 关节含 RIGMESH 占位, 不映射自动忽略), 单位 cm, 100fps;
  2. source_tpose = 骨架 zero pose (Character1 offsets 即标准 T-pose, 无需首帧);
  3. scale = 0.011 (cm->m x ~1.1 骨架放大, 见 data/configs/patient_bvh_to_bio.yaml);
  4. 无合成 T-pose 需剔除。

用法 (仓库根目录):
    PYTHONPATH=. python data/scripts/convert_patient_bvh_to_bio.py \
        --input data/patient_bvh/splits --output data/patient_bio_npy \
        --patients Patient_001  # 可选, 默认全部
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import yaml
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.scripts.lafan_utils import read_bvh  # noqa: E402
from poselib.skeleton.skeleton3d import (  # noqa: E402
    SkeletonMotion,
    SkeletonState,
    SkeletonTree,
)

CONFIG = os.path.join(REPO_ROOT, "data", "configs", "patient_bvh_to_bio.yaml")
BIO_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "mjcf", "bio.xml")


def _as_numpy(x):
    return x.detach().cpu().numpy()


def _rotate_global(m: SkeletonMotion, rot: R):
    """全局旋转所有关节 (y-up -> z-up: 绕 x +90°)。"""
    gq = _as_numpy(m.global_rotation)                     # (T,J,4) xyzw
    gq_flat = gq.reshape(-1, 4)
    q_new = (rot * R.from_quat(gq_flat[:, :4], scalar_first=False)).as_quat(canonical=False)
    st = SkeletonState.from_rotation_and_root_translation(
        m.skeleton_tree,
        torch.from_numpy(q_new.reshape(gq.shape)).float(),
        m.root_translation,
        is_local=False,
    )
    return SkeletonMotion.from_skeleton_state(st, fps=float(m.fps))


def _foot_on_ground(m: SkeletonMotion):
    """整体平移使最低关节 (脚底) z = 0。"""
    min_z = float(_as_numpy(m.global_translation)[..., 2].min())
    if abs(min_z) > 1e-4:
        rt = _as_numpy(m.root_translation) + np.array([0.0, 0.0, -min_z])
        st = SkeletonState.from_rotation_and_root_translation(
            m.skeleton_tree, m.global_rotation, torch.from_numpy(rt).float(),
            is_local=False)
        return SkeletonMotion.from_skeleton_state(st, fps=float(m.fps))
    return m


def _lowpass(x: np.ndarray, fs: float, cutoff: float):
    """2 阶 Butterworth 低通 (零相位 filtfilt), 去除动捕根位置高频噪声。"""
    b, a = butter(2, cutoff / (fs / 2.0), btype="low")
    return np.ascontiguousarray(filtfilt(b, a, x, axis=0))


def convert_one(input_path: str, output_path: str, cfg: dict, filter_hz: float = 6.0):
    anim = read_bvh(input_path)

    # 0. 根位置低通滤波 (病人 BVH 根位置含 ±40cm 帧间噪声, 见 validate 报告)
    root_pos = anim.pos[:, 0].copy()
    if filter_hz > 0:
        root_pos = _lowpass(root_pos, fs=float(anim.fps), cutoff=filter_hz)

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
        torch.from_numpy(root_pos).float(),  # cm
        is_local=True,
    )
    src_motion = SkeletonMotion.from_skeleton_state(src_state, fps=float(anim.fps))

    # 2. source_tpose = zero pose (Character1 offsets 即标准 T-pose)
    source_tpose = SkeletonState.zero_pose(src_tree)

    # 3. target = BIO zero pose
    target_tree = SkeletonTree.from_mjcf(BIO_XML)
    target_tpose = SkeletonState.zero_pose(target_tree)

    # 4. retarget
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

    # 5. y-up -> z-up
    new_motion = _rotate_global(new_motion, R.from_euler("x", 90, degrees=True))

    # 6. 脚贴地
    new_motion = _foot_on_ground(new_motion)

    # 7. 100fps -> 30fps 重采样
    fps_src = round(anim.fps)
    target_fps = 30
    if fps_src != target_fps:
        T_src = int(new_motion.global_rotation.shape[0])
        indices = np.round(np.linspace(0, T_src - 1, int(T_src * target_fps / fps_src))).astype(int)
        sliced = SkeletonState.from_rotation_and_root_translation(
            new_motion.skeleton_tree,
            new_motion.global_rotation[indices],
            new_motion.root_translation[indices],
            is_local=False,
        )
        new_motion = SkeletonMotion.from_skeleton_state(sliced, fps=target_fps)

    # 8. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_motion.to_file(output_path)
    return new_motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(REPO_ROOT, "data", "patient_bvh", "splits"))
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "data", "patient_bio_npy"))
    parser.add_argument("--patients", nargs="+", default=None)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--filter-hz", type=float, default=6.0,
                        help="根位置低通截止频率 (Hz), 0=不过滤; 源数据帧间噪声大, 默认 6Hz")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"配置: scale={cfg['source_length_scale']}, mapping={len(cfg['joint_mapping'])} 关节")

    files = sorted(glob.glob(os.path.join(args.input, "*.bvh")))
    if args.patients:
        files = [f for f in files if any(p in os.path.basename(f) for p in args.patients)]
    print(f"片段: {len(files)}")

    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        pid = base.split("_")[0] + "_" + base.split("_")[1]
        out = os.path.join(args.output, pid, base + ".npy")
        m = convert_one(f, out, cfg, filter_hz=args.filter_hz)
        print(f"{base}: {m.global_rotation.shape[0]} 帧 -> {out}")


if __name__ == "__main__":
    main()
