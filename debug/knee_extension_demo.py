"""Level 5: knee extension/flexion 单关节 biomechanical validation。

固定 pelvis(fix_base_link)+ 其余关节(PD 保持 0),仅允许左膝(L_Knee)运动。
  Phase 1: 激活 left quadriceps -> 观察 L_Knee 向 extension 方向(角度减小)
  Phase 2: 激活 left hamstrings  -> 观察 L_Knee 向 flexion 方向(角度增大)

输出: 控制台结果 + output/knee_demo.png 曲线 + output/knee_demo_traj.npz 轨迹。

运行方式(仓库根目录):
    PYTHONPATH=. python debug/knee_extension_demo.py
"""

import os
import sys

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
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
OUT_DIR = os.path.join(REPO_ROOT, "output")


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


def knee_spawn_cfg():
    """BIO_ACT_CFG 的 spawn 配置 + disable_gravity=True(避免自由骨盆下坠)。"""
    return sim_utils.UsdFileCfg(
        usd_path="protomotions/data/assets/usd/bio.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # 关闭重力,隔离单关节运动
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    )


@configclass
class KneeSceneCfg(InteractiveSceneCfg):
    env_spacing: float = 0.0

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot: ArticulationCfg = BIO_ACT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators=build_actuators(),
        spawn=knee_spawn_cfg(),
    )


def select_muscle_indices(names, prefix="L_", subs=()):
    return [i for i, n in enumerate(names) if n.startswith(prefix) and any(s in n for s in subs)]


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim = SimulationContext(sim_utils.SimulationCfg(device=device, dt=1.0 / 240.0))
    scene = InteractiveScene(KneeSceneCfg(num_envs=1))
    sim.reset()
    robot: Articulation = scene["robot"]
    dt = 1.0 / 240.0

    sim_bodies = list(robot.data.body_names)
    sim_joints = list(robot.data.joint_names)
    body_convert = torch.tensor([sim_bodies.index(n) for n in BODY_NAMES], dtype=torch.long, device=device)
    dof_convert = torch.tensor([sim_joints.index(n) for n in DOF_NAMES], dtype=torch.long, device=device)
    dof_convert_to_sim = torch.tensor([DOF_NAMES.index(n) for n in sim_joints], dtype=torch.long, device=device)

    ctl = MuscleController(muscle_xml_path=MUSCLE_XML, rig_path=BIO_XML, device=torch.device(device))
    ctl.prepare(BODY_NAMES, DOF_NAMES)
    names = [m.name for m in ctl.muscle_char.muscles]

    quad_idx = select_muscle_indices(names, "L_", ("Rectus_Femoris", "Vastus"))
    ham_idx = select_muscle_indices(names, "L_", ("Semimembranosus", "Semitendinosus", "Gracilis", "Sartorius", "Popliteus"))
    print(f"L quadriceps ({len(quad_idx)}): {[names[i] for i in quad_idx]}")
    print(f"L hamstrings ({len(ham_idx)}): {[names[i] for i in ham_idx]}")

    l_knee_common = DOF_NAMES.index("L_Knee")
    l_knee_sim = sim_joints.index("L_Knee")

    # 保持增益:除 L_Knee 外全部关节用 PD 锁在 0
    kp_hold = torch.full((1, len(sim_joints)), 500.0, device=device)
    kd_hold = torch.full((1, len(sim_joints)), 10.0, device=device)
    kp_hold[0, l_knee_sim] = 0.0
    kd_hold[0, l_knee_sim] = 0.0

    def get_common_state_and_features():
        body_pos = robot.data.body_pos_w.clone()[:, body_convert, :]
        body_rot = wxyz_to_xyzw(robot.data.body_quat_w.clone())[:, body_convert, :]
        com = robot.data.body_com_pos_w.clone()[:, body_convert, :]
        jac = robot.root_physx_view.get_jacobians()
        if jac.shape[-1] != len(DOF_NAMES):
            jac = jac[..., -len(DOF_NAMES):]
        jac = jac[:, body_convert, :, :][:, :, :, dof_convert]
        JtA, b = ctl.update_muscle_features(body_pos, body_rot, com, jac)
        return JtA, b

    def set_knee_angle(angle):
        # 通过写关节状态把 L_Knee 设为目标角,其余保持 0
        q = robot.data.joint_pos.clone()
        q[0, l_knee_sim] = angle
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))

    def run_phase(name, act_idx, init_angle, steps=360, activation=0.3):
        set_knee_angle(init_angle)
        a = torch.zeros(1, len(names), device=device)
        a[0, act_idx] = activation
        log = {"t": [], "q": [], "qd": [], "tau": [], "force": []}
        for s in range(steps):
            JtA, b = get_common_state_and_features()
            tau_muscle = ctl._compute_torque(a, JtA, b)  # (1,50) common
            # 主动力矩在 L_Knee 上的贡献(示意,即 aᵀ·JtA 的第 L_Knee 分量)
            force = (a.unsqueeze(-1) * JtA).sum(dim=1)[0, l_knee_common].item()
            q = robot.data.joint_pos  # (1, 50) sim order
            qd = robot.data.joint_vel
            tau_hold = -kp_hold * q - kd_hold * qd
            tau_total = tau_hold.clone()
            tau_total[0, l_knee_sim] = tau_muscle[0, l_knee_common]
            robot.set_joint_effort_target(tau_total)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            log["t"].append(s * dt)
            log["q"].append(q[0, l_knee_sim].item())
            log["qd"].append(qd[0, l_knee_sim].item())
            log["tau"].append(tau_muscle[0, l_knee_common].item())
            log["force"].append(force)
        return {k: np.array(v) for k, v in log.items()}

    # Phase 1: 屈膝 1.0 rad 起,股四头肌激活 -> 应伸膝(角度减小)
    log_ext = run_phase("extension", quad_idx, init_angle=1.0, steps=360, activation=0.3)
    # Phase 2: 近伸直 0.1 rad 起,腘绳肌激活 -> 应屈膝(角度增大)
    log_flex = run_phase("flexion", ham_idx, init_angle=0.1, steps=360, activation=0.3)

    q0_ext, q1_ext = log_ext["q"][0], log_ext["q"][-1]
    q0_flex, q1_flex = log_flex["q"][0], log_flex["q"][-1]
    d_ext = q1_ext - q0_ext
    d_flex = q1_flex - q0_flex

    print("=" * 80)
    print("Level 5 — knee extension/flexion 验证")
    print("=" * 80)
    print(f"Phase 1 (quadriceps, 初始 {q0_ext:.3f} rad): 最终 {q1_ext:.3f} rad, Δ = {d_ext:+.3f} rad")
    print(f"Phase 2 (hamstrings, 初始 {q0_flex:.3f} rad): 最终 {q1_flex:.3f} rad, Δ = {d_flex:+.3f} rad")

    ok_ext = d_ext < -0.05  # 伸膝 = 角度减小
    ok_flex = d_flex > +0.05  # 屈膝 = 角度增大
    print(f"[{'PASS' if ok_ext else 'FAIL'}] quadriceps -> knee extension {'✓' if ok_ext else '✗'}")
    print(f"[{'PASS' if ok_flex else 'FAIL'}] hamstrings  -> knee flexion  {'✓' if ok_flex else '✗'}")

    # 保存轨迹 + 绘图
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "knee_demo_traj.npz"), ext=log_ext, flex=log_flex)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        axes[0].plot(log_ext["t"], log_ext["q"], label="quadriceps (ext)", color="tab:red")
        axes[0].plot(log_flex["t"], log_flex["q"], label="hamstrings (flex)", color="tab:blue")
        axes[0].set_ylabel("L_Knee angle (rad)")
        axes[0].legend()
        axes[1].plot(log_ext["t"], log_ext["qd"], color="tab:red")
        axes[1].plot(log_flex["t"], log_flex["qd"], color="tab:blue")
        axes[1].set_ylabel("L_Knee vel (rad/s)")
        axes[2].plot(log_ext["t"], log_ext["tau"], color="tab:red")
        axes[2].plot(log_flex["t"], log_flex["tau"], color="tab:blue")
        axes[2].set_ylabel("L_Knee torque (N·m)")
        axes[2].set_xlabel("time (s)")
        plt.tight_layout()
        png = os.path.join(OUT_DIR, "knee_demo.png")
        plt.savefig(png, dpi=110)
        print(f"saved plot: {png}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] matplotlib 绘图失败: {e}")

    print("RESULT: " + ("PASS" if (ok_ext and ok_flex) else "FAIL"))
    os._exit(0 if (ok_ext and ok_flex) else 1)


if __name__ == "__main__":
    main()
