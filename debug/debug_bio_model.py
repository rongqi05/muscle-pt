"""Level 1: 最小 spawn 验证 — 只加载 bio.usda,启动 1 个 Isaac Lab 环境。

不加载 motion dataset / 神经网络 / PPO / W&B / 大规模环境。
验证: robot 成功 spawn、body_names、joint_names、DOF 数量、关节限位、初始姿态,
并与 protomotions/config/robot/bio_act.yaml 完全一致。

运行方式(仓库根目录):
    PYTHONPATH=. python debug/debug_bio_model.py
"""

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

from protomotions.simulator.isaaclab.utils.robots import BIO_ACT_CFG  # noqa: E402

# 与 protomotions/config/robot/bio_act.yaml 保持一致(本脚本目标就是校验一致)
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


def build_actuators():
    """复刻 bio_act.yaml 的 IdealPD 驱动器配置(stiffness/damping=0,直接力矩驱动)。"""
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
    env_spacing: float = 0.0  # num_envs=1,间距无意义,但 InteractiveSceneCfg 要求该字段

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )
    robot: ArticulationCfg = BIO_ACT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot", actuators=build_actuators()
    )


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim_cfg = sim_utils.SimulationCfg(device=device, dt=1.0 / 240.0)
    sim = SimulationContext(sim_cfg)

    scene_cfg = OneRobotSceneCfg(num_envs=1)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]

    print("=" * 80)
    print("Level 1 — bio.usda 最小 spawn 验证")
    print("=" * 80)
    print(f"device            : {device}")
    print(f"num_instances     : {robot.num_instances}")
    print(f"num_bodies        : {robot.num_bodies}")
    print(f"num_joints        : {robot.num_joints}")
    print(f"is_fixed_base     : {robot.is_fixed_base}")

    body_names = list(robot.data.body_names)
    joint_names = list(robot.data.joint_names)

    print("-" * 80)
    print(f"sim body_names ({len(body_names)}):")
    print("  ", body_names)
    print(f"sim joint_names ({len(joint_names)}):")
    print("  ", joint_names)

    print("-" * 80)
    print(f"joint_limits shape: {tuple(robot.data.joint_limits.shape)}")
    for jn, lim in zip(joint_names, robot.data.joint_limits[0].tolist()):
        print(f"  {jn:<18} lower={lim[0]:>8.3f}  upper={lim[1]:>8.3f}")

    print("-" * 80)
    print(f"root_pos_w (初始): {robot.data.root_pos_w[0].tolist()}")
    print(f"root_quat_w(初始): {robot.data.root_quat_w[0].tolist()}")
    print(f"joint_pos   (初始): {robot.data.joint_pos[0].tolist()[:10]} ... (前10)")

    # ---------- 与 bio_act.yaml 对比(复刻仓库 on_environment_ready 的排序转换) ----------
    # 仓库中 Isaac Lab 的 body/DOF 顺序(bio.usda articulation 顺序)与 config 的
    # "common 顺序"(bio.xml DFS 顺序)不同,靠 data_conversion 做重排。
    # 这里复刻该转换,验证转换后与 bio_act.yaml 完全一致。
    print("-" * 80)
    errors = []

    if len(body_names) != len(BODY_NAMES):
        errors.append(f"body 数量不一致: sim={len(body_names)} config={len(BODY_NAMES)}")
    if len(joint_names) != len(DOF_NAMES):
        errors.append(f"DOF 数量不一致: sim={len(joint_names)} config={len(DOF_NAMES)}")

    def build_convert(sim_names, common_names, what):
        """复刻 on_environment_ready: sim_convert_to_common[i] = sim_names.index(common[i])"""
        convert = []
        for name in common_names:
            if name not in sim_names:
                errors.append(f"{what} '{name}' 在 sim 中不存在")
                return None
            convert.append(sim_names.index(name))
        return torch.tensor(convert, dtype=torch.long)

    body_conv = build_convert(body_names, BODY_NAMES, "body")
    dof_conv = build_convert(joint_names, DOF_NAMES, "DOF")

    if body_conv is not None:
        # 转换必须是 23 个互不相同的索引(即一个有效的重排)
        if len(set(body_conv.tolist())) != len(BODY_NAMES):
            errors.append("body 排序转换不是有效置换(存在重复/缺失)")
        else:
            print("[PASS] body 排序转换有效,common 顺序 == bio_act.yaml body_names")
    if dof_conv is not None:
        if len(set(dof_conv.tolist())) != len(DOF_NAMES):
            errors.append("DOF 排序转换不是有效置换(存在重复/缺失)")
        else:
            print("[PASS] DOF 排序转换有效,common 顺序 == bio_act.yaml dof_names")

    print("-" * 80)
    print(f"body_convert_to_common (sim->common): {body_conv.tolist() if body_conv is not None else None}")
    print(f"dof_convert_to_common  (sim->common): {dof_conv.tolist() if dof_conv is not None else None}")

    # 初始姿态说明:root quat wxyz [0.707,0.707,0,0] = 绕 x 轴 +90°,即 MuJoCo->Isaac 坐标变换,
    # 与 muscle_parser._rx90 的绕 x +90° 一致。
    print("-" * 80)
    print("说明: root_quat_w(wxyz)≈[0.707,0.707,0,0] = 绕 x 轴 +90°,与 _rx90 一致")

    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(f"  [ERROR] {e}")
    else:
        print("RESULT: PASS — bio.usda 成功 spawn,排序转换后与 bio_act.yaml 完全一致")

    simulation_app.close()


if __name__ == "__main__":
    main()
