"""Level 6b: PD walking — 程序化步态 → PD humanoid 跟踪(不依赖外部数据)。

加载 output/procedural_gait.npz(dof_pos/root_pos/root_rot),
在 Isaac Lab 中 spawn bio humanoid,根节点同步 + 关节 PD 跟踪,
记录轨迹供离线渲染。

运行: PYTHONPATH=. python debug/walk_pd_gait.py
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

from isaac_utils import rotations  # noqa: E402
from protomotions.simulator.isaaclab.utils.robots import BIO_ACT_CFG  # noqa: E402

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

STIFFNESS = {"Hip": 400, "Knee": 300, "Ankle": 400, "Torso": 500, "Spine": 300,
             "Shoulder": 200, "Elbow": 150, "ForeArm": 80, "Hand": 50, "Head": 60,
             "Neck": 60, "Toe": 40}
DAMPING = {"Hip": 6.0, "Knee": 4.5, "Ankle": 6.0, "Torso": 7.5, "Spine": 4.5,
           "Shoulder": 3.0, "Elbow": 2.25, "ForeArm": 1.2, "Hand": 0.75, "Head": 0.9,
           "Neck": 0.9, "Toe": 0.6}


def build_pd_gains():
    kp, kd = [], []
    for name in DOF_NAMES:
        for key in STIFFNESS:
            if key in name:
                kp.append(STIFFNESS[key]); kd.append(DAMPING[key]); break
        else:
            kp.append(0.0); kd.append(0.0)
    return torch.tensor(kp), torch.tensor(kd)


def build_actuators():
    return {
        DOF_NAMES[i]: IdealPDActuatorCfg(
            joint_names_expr=[DOF_NAMES[i]], effort_limit=1000.0,
            velocity_limit=100.0, stiffness=0, damping=0, armature=0.03, friction=0.03,
        )
        for i in range(len(DOF_NAMES))
    }


@configclass
class WalkSceneCfg(InteractiveSceneCfg):
    env_spacing: float = 0.0
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot: ArticulationCfg = BIO_ACT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot", actuators=build_actuators()
    )


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gait = np.load(os.path.join(REPO_ROOT, "output", "procedural_gait.npz"))
    dof_pos_ref = torch.from_numpy(gait["dof_pos"]).to(device)  # (T,50) common
    root_pos_ref = torch.from_numpy(gait["root_pos"]).to(device)  # (T,3)
    root_rot_ref = torch.from_numpy(gait["root_rot"]).to(device)  # (T,4) xyzw
    T = dof_pos_ref.shape[0]
    fps = 30.0

    sim = SimulationContext(sim_utils.SimulationCfg(device=device, dt=1.0 / 240.0))
    scene = InteractiveScene(WalkSceneCfg(num_envs=1))
    sim.reset()
    robot: Articulation = scene["robot"]
    dt = 1.0 / 240.0

    sim_joints = list(robot.data.joint_names)
    dof_to_sim = torch.tensor([DOF_NAMES.index(n) for n in sim_joints], dtype=torch.long, device=device)
    kp_c, kd_c = build_pd_gains()
    kp_sim = kp_c[dof_to_sim.cpu()].to(device).unsqueeze(0)
    kd_sim = kd_c[dof_to_sim.cpu()].to(device).unsqueeze(0)

    root_vel = torch.zeros_like(root_pos_ref)
    if T > 1:
        root_vel[1:] = (root_pos_ref[1:] - root_pos_ref[:-1]) * fps

    # 初始状态
    q0 = torch.zeros(1, len(sim_joints), device=device)
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    init_root = torch.cat([
        root_pos_ref[0].unsqueeze(0),
        rotations.xyzw_to_wxyz(root_rot_ref[0].unsqueeze(0)),
        root_vel[0].unsqueeze(0),
        torch.zeros(1, 3, device=device),
    ], dim=-1)
    robot.write_root_state_to_sim(init_root)

    substeps = max(1, round(240.0 / fps))  # 8
    log = {"root_pos": [], "root_rot": [], "q": [], "q_ref": []}

    for t in range(T):
        root_state = torch.cat([
            root_pos_ref[t].unsqueeze(0),
            rotations.xyzw_to_wxyz(root_rot_ref[t].unsqueeze(0)),
            root_vel[t].unsqueeze(0),
            torch.zeros(1, 3, device=device),
        ], dim=-1)
        robot.write_root_state_to_sim(root_state)

        q_target_common = dof_pos_ref[t].to(device)
        q_target_sim = q_target_common[dof_to_sim]
        for _ in range(substeps):
            q = robot.data.joint_pos
            qd = robot.data.joint_vel
            tau = kp_sim * (q_target_sim.unsqueeze(0) - q) - kd_sim * qd
            robot.set_joint_effort_target(tau)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

        log["root_pos"].append(robot.data.root_pos_w[0].detach().cpu().numpy())
        log["root_rot"].append(robot.data.root_quat_w[0].detach().cpu().numpy())  # wxyz
        log["q"].append(robot.data.joint_pos[0].detach().cpu().numpy())  # sim order
        log["q_ref"].append(q_target_common.cpu().numpy())

    out = os.path.join(REPO_ROOT, "output", "walk_pd_traj.npz")
    np.savez(out,
             root_pos=np.stack(log["root_pos"]),
             root_rot=np.stack(log["root_rot"]),
             q=np.stack(log["q"]),
             q_ref=np.stack(log["q_ref"]),
             sim_joints=np.array(sim_joints))
    print(f"recorded {T} frames -> {out}")

    order = torch.argsort(dof_to_sim.cpu()).numpy()
    q_common = np.stack(log["q"])[:, order]
    err = np.abs(q_common - np.stack(log["q_ref"])).mean()
    print(f"mean |q - q_ref| = {err:.4f} rad")
    print("RESULT: " + ("PASS" if err < 0.3 else "WARN (跟踪误差偏大)"))
    os._exit(0)


if __name__ == "__main__":
    main()
