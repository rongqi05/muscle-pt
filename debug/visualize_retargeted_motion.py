#!/usr/bin/env python3
"""在 Isaac Lab 中可视化验证重定向后的 BIO 参考运动。

用法 (仓库根目录):
    conda activate env_isaaclab
    python debug/visualize_retargeted_motion.py --motion data/cmu_bio_npy/008/08_01.npy

两种播放模式:
  --mode kinematic  逐帧直接写关节/根状态 (纯姿态回放, 看穿透/扭斜)
  --mode pd         PD 跟踪物理仿真 (看物理可行性与脚滑动)

参数:
  --motion      .npy SkeletonMotion 文件路径
  --mode        kinematic | pd (默认 kinematic)
  --headless    无窗口运行 (WSL2 无 GUI 时用)
  --loop N      循环播放 N 次 (默认 3)
  --slow N      每帧渲染步数 (调慢播放, 默认 1)
"""
import argparse

# 必须在 import isaaclab 之前解析 --headless
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--headless", action="store_true")
_pre_args, _ = _pre_parser.parse_known_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": _pre_args.headless})
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
import glob  # noqa: E402

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

from isaac_utils import rotations, torch_utils  # noqa: E402
from poselib.skeleton.skeleton3d import SkeletonMotion  # noqa: E402
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

DOF_BODY_IDS = list(range(1, 23))
JOINT_AXIS = ['xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'y', 'xyz', 'xyz', 'xyz',
              'y', 'xyz', 'xyz', 'x', 'xyz', 'x', 'x', 'xyz', 'x', 'xyz', 'x', 'x']
_off = 0
DOF_OFFSETS = [0]
for ax in JOINT_AXIS:
    _off += len(ax)
    DOF_OFFSETS.append(_off)

STIFFNESS = {"Hip": 400, "Knee": 300, "Ankle": 400, "Torso": 500, "Spine": 300,
             "Shoulder": 200, "Elbow": 150, "ForeArm": 80, "Hand": 50, "Head": 60,
             "Neck": 60, "Toe": 40}
DAMPING = {"Hip": 6.0, "Knee": 4.5, "Ankle": 6.0, "Torso": 7.5, "Spine": 4.5,
           "Shoulder": 3.0, "Elbow": 2.25, "ForeArm": 1.2, "Hand": 0.75, "Head": 0.9,
           "Neck": 0.9, "Toe": 0.6}


def build_pd_gains():
    kp, kd = [], []
    for name in DOF_NAMES:
        found = False
        for key in STIFFNESS:
            if key in name:
                kp.append(STIFFNESS[key]); kd.append(DAMPING[key]); found = True
                break
        if not found:
            kp.append(0.0); kd.append(0.0)
    return torch.tensor(kp, dtype=torch.float), torch.tensor(kd, dtype=torch.float)


def local_rotation_to_dof(local_rot):
    T = local_rot.shape[0]
    dof_pos = torch.zeros((T, 50), dtype=torch.float)
    for j in range(len(DOF_BODY_IDS)):
        body_id = DOF_BODY_IDS[j]
        off = DOF_OFFSETS[j]
        size = DOF_OFFSETS[j + 1] - off
        joint_q = local_rot[:, body_id]
        if size == 3:
            x, y, z = rotations.get_euler_xyz(joint_q, w_last=True)
            dof_pos[:, off:off + 3] = torch.stack([x, y, z], dim=-1)
        elif size == 1:
            theta, axis = torch_utils.quat_to_angle_axis(joint_q, w_last=True)
            cfg = JOINT_AXIS[j]
            if cfg == "x":
                axis = axis[..., 0]
            elif cfg == "y":
                axis = axis[..., 1]
            elif cfg == "z":
                axis = axis[..., 2]
            dof_pos[:, off] = rotations.normalize_angle(theta * axis)
    return dof_pos


def build_actuators():
    return {
        DOF_NAMES[i]: IdealPDActuatorCfg(
            joint_names_expr=[DOF_NAMES[i]],
            effort_limit=1000.0, velocity_limit=100.0,
            stiffness=0, damping=0, armature=0.03, friction=0.03,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=str, required=True, help=".npy 文件或通配")
    parser.add_argument("--mode", type=str, default="kinematic", choices=["kinematic", "pd"])
    parser.add_argument("--headless", action="store_true", help="无窗口运行")
    parser.add_argument("--loop", type=int, default=3, help="循环播放次数")
    parser.add_argument("--slow", type=int, default=1, help="每帧渲染步数(慢放)")
    args = parser.parse_args()

    files = sorted(glob.glob(args.motion))
    assert files, f"未找到: {args.motion}"
    motion = SkeletonMotion.from_file(files[0])
    print(f"motion: {files[0]}  frames={motion.global_rotation.shape[0]} fps={motion.fps}")

    local_rot = motion.local_rotation  # (T,23,4) xyzw
    dof_pos_ref = local_rotation_to_dof(local_rot)  # (T,50)
    root_pos_ref = motion.root_translation  # (T,3)
    root_rot_ref = motion.global_rotation[:, 0]  # (T,4) xyzw
    T = dof_pos_ref.shape[0]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim = SimulationContext(sim_utils.SimulationCfg(device=device, dt=1.0 / 240.0))
    scene = InteractiveScene(WalkSceneCfg(num_envs=1))
    sim.reset()
    robot: Articulation = scene["robot"]
    dt = 1.0 / 240.0

    sim_joints = list(robot.data.joint_names)
    dof_to_sim = torch.tensor([DOF_NAMES.index(n) for n in sim_joints], dtype=torch.long, device=device)
    kp_common, kd_common = build_pd_gains()
    kp_common = kp_common.to(device)
    kd_common = kd_common.to(device)
    kp_sim = kp_common[dof_to_sim].unsqueeze(0)
    kd_sim = kd_common[dof_to_sim].unsqueeze(0)

    # 初始状态
    q0 = dof_pos_ref[0].to(device)[dof_to_sim]
    robot.write_joint_state_to_sim(q0.unsqueeze(0), torch.zeros_like(q0.unsqueeze(0)))
    init_root = torch.cat([
        root_pos_ref[0].unsqueeze(0).to(device),
        rotations.xyzw_to_wxyz(root_rot_ref[0].unsqueeze(0)).to(device),
        torch.zeros(1, 3, device=device),
        torch.zeros(1, 3, device=device),
    ], dim=-1)
    robot.write_root_state_to_sim(init_root)

    fps = float(motion.fps)
    substeps = max(1, round(240.0 / fps))  # 30fps -> 8 substeps

    print(f"播放模式: {args.mode}  (headless={args.headless}, loop={args.loop})")
    print("按 Ctrl+C 停止。观察: 1)手臂/腿是否穿透躯干 2)躯干是否扭斜 3)支撑脚是否滑动")

    total = 0
    try:
        for _ in range(args.loop):
            for t in range(T):
                root_state = torch.cat([
                    root_pos_ref[t].unsqueeze(0).to(device),
                    rotations.xyzw_to_wxyz(root_rot_ref[t].unsqueeze(0)).to(device),
                    torch.zeros(1, 3, device=device),
                    torch.zeros(1, 3, device=device),
                ], dim=-1)
                robot.write_root_state_to_sim(root_state)

                q_target_common = dof_pos_ref[t].to(device)
                q_target_sim = q_target_common[dof_to_sim]

                for _ in range(args.slow):
                    if args.mode == "pd":
                        for _ in range(substeps):
                            q = robot.data.joint_pos
                            qd = robot.data.joint_vel
                            tau = kp_sim * (q_target_sim.unsqueeze(0) - q) - kd_sim * qd
                            robot.set_joint_effort_target(tau)
                            scene.write_data_to_sim()
                            sim.step()
                            scene.update(dt)
                    else:
                        robot.write_joint_state_to_sim(q_target_sim.unsqueeze(0), torch.zeros_like(q_target_sim.unsqueeze(0)))
                        scene.write_data_to_sim()
                        sim.step(render=not args.headless)
                        scene.update(dt)
                total += 1
    except KeyboardInterrupt:
        pass
    print(f"播放结束, 共渲染 {total} 帧")


if __name__ == "__main__":
    main()
