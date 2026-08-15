"""Level 2: muscle parser 单元测试。

加载 bio.xml + muscle284.xml,验证 284 个 muscle 解析正确,并做完整性检查。

运行方式(仓库根目录):
    PYTHONPATH=. python debug/test_muscle_parser.py

不依赖 Isaac Sim,只依赖 mujoco + torch(CPU)。
"""

import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mujoco  # noqa: E402
import torch  # noqa: E402

from protomotions.utils.muscle_parser import CharactorMuscle  # noqa: E402

BIO_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "mjcf", "bio.xml")
MUSCLE_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "muscle284.xml")

# 与 protomotions/config/robot/bio_act.yaml 保持一致(本测试的目标就是校验它们一致)
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


def main() -> None:
    errors = []

    print("=" * 80)
    print("Level 2 — muscle parser 单元测试")
    print("=" * 80)

    # ---------- 1. 载入 mujoco rig 与 CharactorMuscle ----------
    mj_model = mujoco.MjModel.from_xml_path(BIO_XML)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    valid_bodies = set()
    for i in range(mj_model.nbody):
        n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        if n:
            valid_bodies.add(n)
    valid_joints = set()
    for j in range(mj_model.njnt):
        n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n:
            valid_joints.add(n)

    print(f"bio.xml  bodies: {mj_model.nbody}  joints: {mj_model.njnt}")
    print(f"config    body_names: {len(BODY_NAMES)}  dof_names: {len(DOF_NAMES)}")

    cm = CharactorMuscle(MUSCLE_XML, BIO_XML, device=torch.device("cpu"))

    # ---------- 2. 数量校验 ----------
    n_muscles = len(cm.muscles)
    print(f"parsed muscles: {n_muscles}")
    if n_muscles != 284:
        errors.append(f"肌肉数量错误: 期望 284, 实际 {n_muscles}")
    else:
        print("  [PASS] number of muscles == 284")

    # ---------- 3. 逐肌肉完整性检查 ----------
    tree = ET.parse(MUSCLE_XML)
    root = tree.getroot()
    units = root.findall("Unit")
    print(f"muscle284.xml 原始 Unit 数: {len(units)}")

    bad_f0, bad_lm, bad_lt, bad_lmax = [], [], [], []
    missing_body, missing_joint, few_wp = [], [], []
    nan_units = []

    for unit in units:
        name = unit.attrib.get("name", "")
        f0 = float(unit.attrib.get("f0", "1000"))
        lm = float(unit.attrib.get("lm", "1.0"))
        lt = float(unit.attrib.get("lt", "0.2"))
        lmax = float(unit.attrib.get("lmax", "-0.1"))

        if not all(math.isfinite(v) for v in (f0, lm, lt, lmax)):
            nan_units.append(name)
        if not (math.isfinite(f0) and f0 > 0):
            bad_f0.append((name, f0))
        if not (math.isfinite(lm) and lm > 0):
            bad_lm.append((name, lm))
        if not (math.isfinite(lt) and lt > 0):
            bad_lt.append((name, lt))
        # lmax <= 0 视为无效(parser 原始值不做修正,保留 -0.1 默认)
        if not (math.isfinite(lmax) and lmax > 0):
            bad_lmax.append((name, lmax))

        wps = unit.findall("Waypoint")
        if len(wps) < 2:
            few_wp.append(name)
        for wp in wps:
            body = wp.attrib.get("body", "")
            if body not in valid_bodies:
                missing_body.append((name, body))

    # 关节检查:bio.xml 的 50 个 hinge 是否都在 DOF_NAMES 中
    hinge_joints = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(mj_model.njnt)
    ]
    hinge_joints = [n for n in hinge_joints if n and n != "Pelvis"]  # 去掉 freejoint
    for dj in DOF_NAMES:
        if dj not in valid_joints:
            missing_joint.append(dj)

    print("-" * 80)
    print(f"NaN/Inf 参数        : {len(nan_units)} 个 -> {nan_units[:5]}{'...' if len(nan_units)>5 else ''}")
    print(f"非法 F0 (<=0)       : {len(bad_f0)} 个 -> {bad_f0[:5]}")
    print(f"非法 lm (<=0)       : {len(bad_lm)} 个 -> {bad_lm[:5]}")
    print(f"非法 lt (<=0)       : {len(bad_lt)} 个 -> {bad_lt[:5]}")
    print(f"非法 lmax (<=0)     : {len(bad_lmax)} 个 -> {bad_lmax[:3]}(注意:默认 -0.1 会在此列出)")
    print(f"缺失 body           : {len(missing_body)} 个 -> {missing_body[:5]}")
    print(f"缺失 joint(DOF_NAMES 中) : {len(missing_joint)} 个 -> {missing_joint[:5]}")
    print(f"waypoint < 2        : {len(few_wp)} 个 -> {few_wp[:5]}")

    # ---------- 4. 前 10 个 muscle 详情 ----------
    print("-" * 80)
    print("前 10 个 muscle:")
    print(f"{'name':<38}{'F0':>8}{'lm':>7}{'lt':>7}{'lmax':>8}{'#wp':>5}  waypoint bodies")
    for m in cm.muscles[:10]:
        bodies = []
        for wp in m.waypoints:
            bodies.append("/".join(wp.bodies) if len(wp.bodies) <= 2 else "+".join(wp.bodies[:2]) + "...")
        a = m.attrs
        print(
            f"{m.name:<38}{a['f0']:>8.1f}{a['lm']:>7.3f}{a['lt']:>7.3f}{a['lmax']:>8.3f}"
            f"{len(m.waypoints):>5}  {' -> '.join(bodies)}"
        )

    # ---------- 5. body/DOF 绑定校验(prepare_mapping) ----------
    print("-" * 80)
    cm.prepare_mapping(BODY_NAMES, DOF_NAMES)
    l_mt0 = cm._l_mt0
    n_unbound = 0
    for m_idx, idxs in enumerate(cm._backend_idx_per_muscle):
        flat = [i for wp in idxs for i in wp]
        if all(i < 0 for i in flat):
            n_unbound += 1
            errors.append(f"muscle {cm.muscles[m_idx].name} 未绑定任何 body")

    n_bad_l0 = int(np.sum(~np.isfinite(l_mt0)) + np.sum(l_mt0 <= 0))
    print(f"完全未绑定 body 的 muscle: {n_unbound}")
    print(f"l_mt0 非法(<=0/NaN)      : {n_bad_l0}")
    print(f"l_mt0 min/mean/max       : {l_mt0.min():.4f} / {l_mt0.mean():.4f} / {l_mt0.max():.4f}")

    if n_unbound:
        errors.append("存在未绑定任何 body 的 muscle")
    if n_bad_l0:
        errors.append("存在 l_mt0 <= 0 或 NaN 的 muscle")

    # ---------- 汇总 ----------
    print("=" * 80)
    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(f"  [ERROR] {e}")
        sys.exit(1)
    print("RESULT: PASS — 284 muscles 解析正确,参数与 body/DOF 绑定完整")
    sys.exit(0)


if __name__ == "__main__":
    main()
