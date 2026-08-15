"""Level 4: 验证 JtA 与 τ = JtAᵀ a + b(核心 milestone)。

在 Isaac Lab 中初始化 1 个 humanoid,调用仓库实际使用的
`MuscleController.update_muscle_features()`,验证:
  - JtA.shape == [1, 284, 50]
  - b.shape   == [1, 50]
  - 数值:min/max/mean/norm,NaN/Inf/all-zero
  - Case A: a=0       -> tau ≈ b
  - Case B: a=1       -> 最大主动力矩
  - Case C: 单条 quadriceps -> 力矩主要作用在合理 knee DOF
  - Case D: 只激活左侧 quadriceps -> 右腿力矩 ≈ 0

复刻 IsaacLabSimulator.update_muscle_features 的数据流:
  body_quat_w(wxyz) --wxyz_to_xyzw--> body_rot(xyzw, common 排序)
  root_physx_view.get_jacobians() --重排 body/dof--> common 排序

运行方式(仓库根目录):
    PYTHONPATH=. python debug/debug_jta.py
"""

import os
import sys

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from isaac_utils.rotations import wxyz_to_xyzw  # noqa: E402
from protomotions.simulator.isaaclab.utils.robots import BIO_ACT_CFG  # noqa: E402
from protomotions.utils.muscle_control import MuscleController  # noqa: E402

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

MUSCLE_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "muscle284.xml")
BIO_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "mjcf", "bio.xml")


def build_actuators():
    return {
        DOF_NAMES[i]: IdealPDActuatorCfg(
            joint_names_expr=[DOF_NAMES[i]],
            effort_limit=1000.0,
            velocity_limit=100.0,
            stiffness=0,
            damping=0,
            armature=0.03,
            friction=0.03,
        )
        for i in range(len(DOF_NAMES))
    }


@configclass
class OneRobotSceneCfg(InteractiveSceneCfg):
    env_spacing: float = 0.0

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0)
    )
    robot: ArticulationCfg = BIO_ACT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot", actuators=build_actuators()
    )


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim = SimulationContext(sim_utils.SimulationCfg(device=device, dt=1.0 / 240.0))
    scene = InteractiveScene(OneRobotSceneCfg(num_envs=1))
    sim.reset()
    robot: Articulation = scene["robot"]

    # ---------- 1. 构建排序转换(复刻 on_environment_ready) ----------
    sim_bodies = list(robot.data.body_names)
    sim_joints = list(robot.data.joint_names)
    body_convert = torch.tensor(
        [sim_bodies.index(n) for n in BODY_NAMES], dtype=torch.long, device=device
    )
    dof_convert = torch.tensor(
        [sim_joints.index(n) for n in DOF_NAMES], dtype=torch.long, device=device
    )

    # ---------- 2. 取 body 状态并转 common 排序(xyzw) ----------
    body_pos = robot.data.body_pos_w.clone()[:, body_convert, :]  # (1,23,3)
    body_quat_w = robot.data.body_quat_w.clone()  # (1, num_sim_bodies, 4) wxyz
    body_quat_w = wxyz_to_xyzw(body_quat_w)[:, body_convert, :]  # (1,23,4) xyzw
    com = robot.data.body_com_pos_w.clone()[:, body_convert, :]  # (1,23,3)

    # ---------- 3. Jacobian(复刻 _get_body_jacobians_common) ----------
    jac = robot.root_physx_view.get_jacobians()
    print(f"raw jacobian shape: {tuple(jac.shape)}")
    if jac.shape[-1] != len(DOF_NAMES):
        if jac.shape[-1] > len(DOF_NAMES):
            jac = jac[..., -len(DOF_NAMES):]
        else:
            raise RuntimeError(f"jacobian DOF 维度过小: {jac.shape}")
    jac = jac[:, body_convert, :, :]
    jac = jac[:, :, :, dof_convert]

    # ---------- 4. 构建 MuscleController 并计算 JtA / b ----------
    ctl = MuscleController(muscle_xml_path=MUSCLE_XML, rig_path=BIO_XML, device=torch.device(device))
    ctl.prepare(BODY_NAMES, DOF_NAMES)
    muscle_names = [m.name for m in ctl.muscle_char.muscles]

    JtA, b = ctl.update_muscle_features(body_pos, body_quat_w, com, jac)

    print("=" * 80)
    print("Level 4 — JtA / b 验证")
    print("=" * 80)
    print(f"JtA.shape: {tuple(JtA.shape)}   (期望 [1, 284, 50])")
    print(f"b.shape  : {tuple(b.shape)}   (期望 [1, 50])")

    errors = []
    if tuple(JtA.shape) != (1, 284, 50):
        errors.append(f"JtA 维度错误: {tuple(JtA.shape)}")
    if tuple(b.shape) != (1, 50):
        errors.append(f"b 维度错误: {tuple(b.shape)}")

    JtA_abs = JtA.abs()
    print("-" * 80)
    print(f"JtA min      : {JtA.min().item():.6f}")
    print(f"JtA max      : {JtA.max().item():.6f}")
    print(f"JtA mean     : {JtA.mean().item():.6f}")
    print(f"JtA norm(F)  : {JtA.norm().item():.4f}")
    print(f"b   min      : {b.min().item():.6f}")
    print(f"b   max      : {b.max().item():.6f}")
    print(f"b   mean     : {b.mean().item():.6f}")
    print(f"b   norm     : {b.norm().item():.4f}")

    if torch.isnan(JtA).any() or torch.isinf(JtA).any():
        errors.append("JtA 含 NaN/Inf")
    if torch.isnan(b).any() or torch.isinf(b).any():
        errors.append("b 含 NaN/Inf")
    if (JtA_abs.max() < 1e-12):
        errors.append("JtA 全零")
    else:
        print("[PASS] JtA 非全零")
    if b.norm() < 1e-12:
        print("[WARN] b 接近全零(被动项在当前姿态下为 0,属正常,见下)")
    else:
        print("[PASS] b 非全零")

    # ---------- 5. τ = JtAᵀ a + b 的数值验证 ----------
    def tau(a):
        return ctl._compute_torque(a, JtA, b)

    print("-" * 80)
    # Case A: a = 0
    a0 = torch.zeros(1, 284, device=device)
    tau_a = tau(a0)
    err_a = (tau_a - b).abs().max().item()
    print(f"Case A (a=0): max|tau - b| = {err_a:.2e}  -> {'PASS' if err_a < 1e-6 else 'FAIL'}")
    if err_a >= 1e-6:
        errors.append(f"Case A 失败: tau != b, err={err_a}")

    # Case B: a = 1
    a1 = torch.ones(1, 284, device=device)
    tau_b = tau(a1)
    print(f"Case B (a=1): |tau| max = {tau_b.abs().max().item():.4f},  |tau| mean = {tau_b.abs().mean().item():.4f}")
    print(f"              tau(前10 DOF) = {tau_b[0,:10].tolist()}")

    # Case C: 单条 quadriceps(L_Rectus_Femoris)
    idx_rf = muscle_names.index("L_Rectus_Femoris")
    a_rf = torch.zeros(1, 284, device=device)
    a_rf[0, idx_rf] = 1.0
    tau_rf = tau(a_rf)[0]
    knee_idx = DOF_NAMES.index("L_Knee")
    hip_idx = DOF_NAMES.index("L_Hip_x")
    print("-" * 80)
    print(f"Case C (仅 L_Rectus_Femoris, idx={idx_rf}):")
    print(f"  L_Knee (idx {knee_idx}) torque = {tau_rf[knee_idx].item():.4f}")
    print(f"  L_Hip_x(idx {hip_idx}) torque = {tau_rf[hip_idx].item():.4f}")
    top3 = torch.topk(tau_rf.abs(), 3)
    print(f"  |tau| top3 DOF: {[(DOF_NAMES[i], f'{v:.4f}') for v, i in zip(top3.values, top3.indices)]}")
    if DOF_NAMES[top3.indices[0]] in ("L_Knee", "L_Hip_x", "L_Hip_y", "L_Hip_z"):
        print("  [PASS] 主要力矩作用在左侧髋/膝(合理)")
    else:
        print("  [WARN] 主要力矩不在预期 DOF,需检查 waypoint/body mapping")
        errors.append(f"Case C: 主要力矩在 {DOF_NAMES[top3.indices[0]]},预期 L_Knee/L_Hip")

    # Case D: 只激活左侧 quadriceps,检查右腿力矩 ≈ 0
    lq_idx = [i for i, n in enumerate(muscle_names) if n.startswith("L_") and ("Rectus_Femoris" in n or "Vastus" in n)]
    a_lq = torch.zeros(1, 284, device=device)
    a_lq[0, lq_idx] = 1.0
    tau_lq = tau(a_lq)[0]
    right_dofs = [DOF_NAMES.index(n) for n in DOF_NAMES if n.startswith("R_")]
    right_tau_max = tau_lq[right_dofs].abs().max().item()
    left_knee = tau_lq[knee_idx].abs().item()
    print("-" * 80)
    print(f"Case D (仅左侧 quadriceps, {len(lq_idx)} 条):")
    print(f"  L_Knee |tau| = {left_knee:.4f}")
    print(f"  右腿 DOF |tau| max = {right_tau_max:.6f}")
    if right_tau_max < 1e-3 * max(left_knee, 1.0):
        print("  [PASS] 右腿力矩 ≈ 0,无 cross-side torque")
    else:
        print("  [FAIL] 存在明显 cross-side torque,需检查 waypoint/body mapping")
        errors.append(f"Case D: 右腿力矩 {right_tau_max:.4f} 过大")

    print("=" * 80)
    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(f"  [ERROR] {e}")
    else:
        print("RESULT: PASS — JtA/b 维度与 τ=JtAᵀa+b 全部验证通过")

    # 注意: headless 下 simulation_app.close() 在调用 get_jacobians() 之后有时会挂起,
    # 所有验证均已在此前完成,这里直接强制退出避免清理挂起。
    os._exit(1 if errors else 0)


if __name__ == "__main__":
    main()
