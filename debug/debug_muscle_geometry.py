"""Level 3: muscle geometry debug(纯 mujoco + numpy,不依赖 Isaac Sim)。

在固定人体姿态下,计算每条肌肉:
  - waypoint 世界坐标
  - 肌肉总长度 l_mt(绝对,米)与归一化长度 l_mt/l_mt0
  - 归一化肌纤维长度 x = l_m / l_m0(复刻 update_muscle_features 的公式)
  - 主动力-长度系数 g_al(x)
  - 被动力 g_pl(x)

重点打印:quadriceps / hamstrings / gastrocnemius / soleus / tibialis anterior / gluteus。
验证: l_m > 0 且所有结果 finite(无 NaN / Inf)。

运行方式(仓库根目录):
    PYTHONPATH=. python debug/debug_muscle_geometry.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mujoco  # noqa: E402
import torch  # noqa: E402

from protomotions.utils.muscle_parser import CharactorMuscle  # noqa: E402

BIO_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "mjcf", "bio.xml")
MUSCLE_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "muscle284.xml")

BODY_NAMES = [
    "Pelvis", "Spine", "Torso", "Neck", "Head",
    "ShoulderL", "ArmL", "ForeArmL", "HandL",
    "ShoulderR", "ArmR", "ForeArmR", "HandR",
    "FemurL", "TibiaL", "TalusL", "FootThumbL", "FootPinkyL",
    "FemurR", "TibiaR", "TalusR", "FootThumbR", "FootPinkyR",
]
DOF_NAMES = [
    "Spine_x", "Spine_y", "Spine_z", "Torso_x", "Torso_y", "Torso_z",
    "Neck_x", "Neck_y", "Neck_z", "Head_x", "Head_y", "Head_z",
    "L_Shoulder_x", "L_Shoulder_y", "L_Shoulder_z", "L_Elbow_x", "L_Elbow_y", "L_Elbow_z",
    "ForeArmL_y", "HandL_x", "HandL_y", "HandL_z",
    "R_Shoulder_x", "R_Shoulder_y", "R_Shoulder_z", "R_Elbow_x", "R_Elbow_y", "R_Elbow_z",
    "ForeArmR_y", "HandR_x", "HandR_y", "HandR_z",
    "L_Hip_x", "L_Hip_y", "L_Hip_z", "L_Knee", "L_Ankle_x", "L_Ankle_y", "L_Ankle_z",
    "L_ToeThumb", "L_ToePinky",
    "R_Hip_x", "R_Hip_y", "R_Hip_z", "R_Knee", "R_Ankle_x", "R_Ankle_y", "R_Ankle_z",
    "R_ToeThumb", "R_ToePinky",
]

# Hill 参数(与 muscle_parser.CharactorMuscle 一致)
GAMMA = 0.5
K_PE = 4.0
E_MO = 0.6

# 关键肌群搜索模式(按 MASS 命名)
KEY_PATTERNS = {
    "quadriceps": ["Rectus_Femoris", "Vastus"],
    "hamstrings": ["Semitendinosus", "Semimembranosus", "Biceps_Femoris"],
    "gastrocnemius": ["Gastrocnemius"],
    "soleus": ["Soleus"],
    "tibialis_anterior": ["Tibialis_Anterior"],
    "gluteus": ["Gluteus"],
}


def compute_geometry(cm, mj_data, pose_name):
    """在给定 mujoco 姿态下,复刻 update_muscle_features 的几何/长度公式。"""
    mujoco.mj_forward(cm.m, mj_data)

    xpos = np.asarray(mj_data.xpos)  # (nbody, 3)
    xmat = np.asarray(mj_data.xmat).reshape(-1, 3, 3)  # (nbody, 3, 3)

    idx = cm._padded_backend_idx  # (M, maxW, maxK) -> BODY_NAMES 索引
    local = cm._padded_local_pts  # (M, maxW, maxK, 3)
    w = cm._padded_weights  # (M, maxW, maxK)

    M, maxW, maxK = idx.shape
    # idx 存的是 BODY_NAMES 顺序索引(0~22),必须映射回 mujoco body id
    # (mujoco: 0=world, 1=Pelvis, ...)。直接拿 idx 当 mujoco 索引会差一位。
    mj_body_ids = np.array([cm.body_name_to_id[n] for n in BODY_NAMES], dtype=np.int64)
    safe = np.clip(idx, 0, None)
    mj_idx = mj_body_ids[safe]  # (M, maxW, maxK) -> mujoco body id

    bp = xpos[mj_idx]  # (M, maxW, maxK, 3)
    bR = xmat[mj_idx]  # (M, maxW, maxK, 3, 3)

    world = bp + np.matmul(bR, local[..., None]).squeeze(-1)  # (M, maxW, maxK, 3)
    p_world = (world * w[..., None]).sum(axis=2)  # (M, maxW, 3)

    wp_mask = w.sum(axis=2) > 0  # (M, maxW)
    seg = p_world[:, 1:, :] - p_world[:, :-1, :]  # (M, maxW-1, 3)
    seg_len = np.linalg.norm(seg, axis=-1)  # (M, maxW-1)
    seg_mask = wp_mask[:, 1:] & wp_mask[:, :-1]

    l_mt_total = (seg_len * seg_mask).sum(axis=1)  # (M,) 绝对总长
    l_mt0 = np.asarray(cm._l_mt0)
    l_mt_norm = l_mt_total / l_mt0  # 归一化(update_muscle_features 的定义)

    l_t0 = cm.l_t0.numpy()
    l_m0 = cm.l_m0.numpy()
    f0 = cm.f0.numpy()

    l_m = l_mt_norm - l_t0  # (M,) 代码定义
    x = l_m / l_m0  # (M,) 归一化肌纤维长度

    g_al = np.exp(-(x - 1.0) ** 2 / GAMMA)
    g_pl_raw = (np.exp(K_PE * (x - 1.0) / E_MO) - 1.0) / (np.exp(K_PE) - 1.0)
    g_pl = np.where(x < 1.0, 0.0, g_pl_raw)

    return {
        "pose": pose_name,
        "p_world": p_world,
        "l_mt_total": l_mt_total,
        "l_mt_norm": l_mt_norm,
        "l_m": l_m,
        "x": x,
        "g_al": g_al,
        "g_pl": g_pl,
        "f_active_scale": f0 * g_al,
        "f_passive": f0 * g_pl,
    }


def main():
    cm = CharactorMuscle(MUSCLE_XML, BIO_XML, device=torch.device("cpu"))
    cm.prepare_mapping(BODY_NAMES, DOF_NAMES)
    names = [m.name for m in cm.muscles]

    # 姿态 1:默认姿态(bio.xml 默认 = 所有关节 0)
    d_default = mujoco.MjData(cm.m)
    geo_default = compute_geometry(cm, d_default, "default(T-pose)")

    # 姿态 2:轻微屈膝(左膝 0.6 rad,右膝 0.4 rad,左髋屈 0.3 rad),观察长度变化
    d_bent = mujoco.MjData(cm.m)
    for jn, ang in [("L_Knee", 0.6), ("R_Knee", 0.4), ("L_Hip_x", 0.3)]:
        jid = cm.joint_name_to_id.get(jn)
        if jid is not None:
            # 铰链关节的 qpos 地址是 jnt_qposadr[jid](前面还有 freejoint 的 7 个分量)
            qadr = cm.m.jnt_qposadr[jid]
            d_bent.qpos[qadr] = ang
    geo_bent = compute_geometry(cm, d_bent, "bent-knee")

    # 全量校验
    errors = []
    for geo in (geo_default, geo_bent):
        if not np.all(np.isfinite(geo["l_m"])):
            errors.append(f"{geo['pose']}: l_m 含 NaN/Inf")
        if not np.all(np.isfinite(geo["x"])):
            errors.append(f"{geo['pose']}: x 含 NaN/Inf")
        if not np.all(np.isfinite(geo["g_al"])) or not np.all(np.isfinite(geo["g_pl"])):
            errors.append(f"{geo['pose']}: Hill 项含 NaN/Inf")
        n_neg = int(np.sum(geo["l_m"] <= 0))
        print(f"[{geo['pose']}] l_m <= 0 的肌肉数: {n_neg}")
        if n_neg > 0:
            errors.append(f"{geo['pose']}: {n_neg} 条肌肉 l_m <= 0")

    # 打印关键肌群
    print("=" * 100)
    for group, pats in KEY_PATTERNS.items():
        print(f"\n### {group}")
        print(f"{'name':<36}{'l_mt_total(m)':>13}{'l_mt/l_mt0':>11}{'x=l_m/l_m0':>11}{'g_al':>8}{'g_pl':>8}  "
              f"{'(bent) l_mt_total':>16}{'(bent) x':>10}")
        for name in names:
            if any(p in name for p in pats):
                i = names.index(name)
                print(
                    f"{name:<36}{geo_default['l_mt_total'][i]:>13.4f}{geo_default['l_mt_norm'][i]:>11.3f}"
                    f"{geo_default['x'][i]:>11.3f}{geo_default['g_al'][i]:>8.3f}{geo_default['g_pl'][i]:>8.3f}  "
                    f"{geo_bent['l_mt_total'][i]:>16.4f}{geo_bent['x'][i]:>10.3f}"
                )

    # 汇总
    print("=" * 100)
    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(f"  [ERROR] {e}")
        sys.exit(1)
    print("RESULT: PASS — 所有肌肉几何量 finite,l_m > 0")
    sys.exit(0)


if __name__ == "__main__":
    main()
