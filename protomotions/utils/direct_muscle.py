"""纯直接优化肌肉控制 (无 RL, 不训练网络)。

技术路线:
    BVH/BIO q_ref -> PD desired torque -> optimize 284 activation
        -> muscle model (Hill) -> 50 torque -> Isaac Lab

这是一个可复用的独立模块, 供 CLI/批量评估/动画生成调用。核心逻辑原本在
debug/walk_muscle_demo.py, 现固化于此。

用法 (必须先创建 SimulationApp, 因为 isaaclab 导入必须在 AppLauncher 之后):

    from isaaclab.app import AppLauncher
    app = AppLauncher({"headless": True}).app

    from protomotions.utils.direct_muscle import DirectMuscleTracker
    tracker = DirectMuscleTracker(device="cuda:0", simulation_app=app, use_scale_map=True)
    res = tracker.track("data/cmu_bio_npy/009/09_12.npy", max_frames=120)
    print(f"tracking_err = {res['tracking_err']:.4f} rad")
    tracker.close()
"""

import os
import numpy as np
import torch

from isaac_utils import rotations, torch_utils
from isaac_utils.rotations import wxyz_to_xyzw
from poselib.skeleton.skeleton3d import SkeletonMotion
from protomotions.utils.muscle_control import MuscleController

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# dof_body_ids / joint_axis 与 bio_act.yaml 一致
DOF_BODY_IDS = list(range(1, 23))
JOINT_AXIS = ['xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'y', 'xyz', 'xyz', 'xyz',
              'y', 'xyz', 'xyz', 'x', 'xyz', 'x', 'x', 'xyz', 'x', 'xyz', 'x', 'x']
_off = 0
DOF_OFFSETS = [0]
for _ax in JOINT_AXIS:
    _off += len(_ax)
    DOF_OFFSETS.append(_off)

# PD 增益 (bio_act.yaml, 按子串匹配)
STIFFNESS = {"Hip": 400, "Knee": 300, "Ankle": 400, "Torso": 500, "Spine": 300,
             "Shoulder": 200, "Elbow": 150, "ForeArm": 80, "Hand": 50, "Head": 60,
             "Neck": 60, "Toe": 40}
DAMPING = {"Hip": 6.0, "Knee": 4.5, "Ankle": 6.0, "Torso": 7.5, "Spine": 4.5,
           "Shoulder": 3.0, "Elbow": 2.25, "ForeArm": 1.2, "Hand": 0.75, "Head": 0.9,
           "Neck": 0.9, "Toe": 0.6}

MUSCLE_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "muscle284.xml")
BIO_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "mjcf", "bio.xml")

# 偏弱肌肉的 f0 补偿倍率 (踝/足/趾 5~20x, 髋 3~6x)
MUSCLE_SCALE_MAP = {
    "L_Extensor_Digitorum_Longus": 12.0, "L_Extensor_Digitorum_Longus1": 12.0,
    "L_Extensor_Digitorum_Longus2": 12.0, "L_Extensor_Digitorum_Longus3": 12.0,
    "L_Flexor_Digitorum_Longus": 12.0, "L_Flexor_Digitorum_Longus1": 12.0,
    "L_Flexor_Digitorum_Longus2": 12.0, "L_Flexor_Digitorum_Longus3": 12.0,
    "L_Flexor_Digiti_Minimi_Brevis_Foot": 12.0, "L_Flexor_Hallucis": 8.0,
    "L_Flexor_Hallucis1": 8.0, "L_Extensor_Hallucis_Longus": 8.0,
    "L_Tibialis_Anterior": 14.0, "L_Tibialis_Posterior": 5.0,
    "L_Peroneus_Longus": 11.0, "L_Peroneus_Brevis": 11.0,
    "L_Peroneus_Tertius": 12.0, "L_Peroneus_Tertius1": 12.0, "L_Plantaris": 12.0,
    "L_Soleus": 11.0, "L_Soleus1": 9.0,
    "L_Gastrocnemius_Medial_Head": 3.0, "L_Gastrocnemius_Lateral_Head": 3.0,
    "L_Semimembranosus": 3.0, "L_Semitendinosus": 2.0,
    "L_Psoas_Major1": 6.0, "L_Psoas_Major2": 4.5, "L_Psoas_Minor": 3.0,
    "L_Pectineus": 4.0, "L_Gluteus_Medius": 4.0, "L_Gluteus_Medius1": 4.0,
    "L_Gluteus_Medius2": 4.0, "L_Gluteus_Medius3": 4.0,
    "L_Gluteus_Maximus1": 3.0, "L_Gluteus_Maximus2": 3.0,
    "L_Gluteus_Maximus3": 3.0, "L_Gluteus_Maximus4": 3.0, "L_Quadratus_Lumborum1": 2.5,
    "R_Extensor_Digitorum_Longus": 14.0, "R_Extensor_Digitorum_Longus1": 14.0,
    "R_Extensor_Digitorum_Longus2": 14.0, "R_Extensor_Digitorum_Longus3": 16.0,
    "R_Flexor_Digitorum_Longus": 20.0, "R_Flexor_Digitorum_Longus1": 20.0,
    "R_Flexor_Digitorum_Longus2": 16.0, "R_Flexor_Digitorum_Longus3": 16.0,
    "R_Flexor_Digiti_Minimi_Brevis_Foot": 12.0, "R_Flexor_Hallucis": 8.0,
    "R_Flexor_Hallucis1": 8.0, "R_Extensor_Hallucis_Longus": 8.0,
    "R_Tibialis_Anterior": 20.0, "R_Tibialis_Posterior": 5.0,
    "R_Peroneus_Longus": 12.0, "R_Peroneus_Brevis": 12.0,
    "R_Peroneus_Tertius": 12.0, "R_Peroneus_Tertius1": 12.0, "R_Plantaris": 12.0,
    "R_Soleus": 8.0, "R_Soleus1": 6.0,
    "R_Gastrocnemius_Medial_Head": 3.0, "R_Gastrocnemius_Lateral_Head": 3.0,
    "R_Semimembranosus": 3.0, "R_Semitendinosus": 2.0,
    "R_Psoas_Major1": 6.0, "R_Psoas_Major2": 4.5, "R_Psoas_Minor": 3.0,
    "R_Pectineus": 4.0, "R_Gluteus_Medius": 4.0, "R_Gluteus_Medius1": 4.0,
    "R_Gluteus_Medius2": 4.0, "R_Gluteus_Medius3": 4.0,
    "R_Gluteus_Maximus1": 3.0, "R_Gluteus_Maximus2": 3.0,
    "R_Gluteus_Maximus3": 3.0, "R_Gluteus_Maximus4": 3.0,
    "R_Latissimus_Dorsi3": 3.0, "R_Serratus_Posterior_Inferior": 3.0,
}


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
    """复刻 motion_lib._local_rotation_to_dof (joint_3d_format='exp_map')。local_rot: (T, 23, 4) xyzw。

    注意: 3 自由度关节必须用 quat_to_exp_map (角度 ∈ [-pi, pi]), 不能用
    get_euler_xyz (后者返回 % 2pi, 会把小负角包成 ~6.28 rad)。
    """
    T = local_rot.shape[0]
    dof_pos = torch.zeros((T, 50), dtype=torch.float)
    for j in range(len(DOF_BODY_IDS)):
        body_id = DOF_BODY_IDS[j]
        off = DOF_OFFSETS[j]
        size = DOF_OFFSETS[j + 1] - off
        joint_q = local_rot[:, body_id]  # (T,4) xyzw
        if size == 3:
            dof_pos[:, off:off + 3] = torch_utils.quat_to_exp_map(joint_q, w_last=True)
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


def optimize_act(JtA, b, tau, method="ls", max_iter=5, last_act=None):
    """解 284 激活 a, 使 tau_pred = a·JtA + b 逼近 tau。

    与 protomotions/simulator/isaaclab/simulator.py::optimize_act 同款数学。
    method: 'ls' (最小二乘初值 + LBFGS 精修) 或 'lbfgs'。
    """
    JtA = JtA.detach()
    b = b.detach()
    tau_target = tau.detach()
    B, M, D = JtA.shape

    def _tau(a):
        return torch.einsum("bm,bmd->bd", a, JtA) + b

    if method == "ls":
        target = (tau_target - b).unsqueeze(-1)  # (B,D,1)
        matrix = JtA.transpose(1, 2)             # (B,D,M)
        a_sol = torch.linalg.lstsq(matrix, target).solution.squeeze(-1)  # (B,M)
        a_clamped = torch.clamp(a_sol, 1e-4, 1.0 - 1e-4)
        x = torch.logit(a_clamped).clone().detach().requires_grad_(True)
    elif method == "lbfgs":
        x = torch.zeros((B, M), device=JtA.device, requires_grad=True)
    else:
        raise ValueError(f"未知优化方法: {method}")

    opt = torch.optim.LBFGS([x], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        a = torch.sigmoid(x)
        loss = (_tau(a) - tau_target).pow(2).sum()
        if last_act is not None:
            loss = loss + 0.05 * (a - last_act).pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)

    with torch.no_grad():
        a_final = torch.sigmoid(x)
        tau_final = _tau(a_final)
    return a_final, tau_final


class DirectMuscleTracker:
    """直接肌肉控制跟踪器: 逐帧反解 284 激活并施加肌肉力矩驱动 BIO 人形。

    内部持有 Isaac Lab 场景 + 机器人 + MuscleController, 可跨 motion 复用。
    """

    def __init__(self, device="cuda:0", simulation_app=None, use_scale_map=True,
                 torque_limit=1000.0, dt=1.0 / 240.0):
        # isaaclab 必须在 AppLauncher 之后才可导入, 故延迟到此处
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.utils import configclass
        from protomotions.simulator.isaaclab.utils.robots import BIO_ACT_CFG

        self.device = device
        self._simulation_app = simulation_app
        self.torque_limit = torque_limit
        self.dt = dt
        self.sim_joints = list(DOF_NAMES)  # 占位, 下面用真实顺序覆盖

        def build_actuators():
            return {
                DOF_NAMES[i]: IdealPDActuatorCfg(
                    joint_names_expr=[DOF_NAMES[i]],
                    effort_limit=1000.0,
                    velocity_limit=100.0,
                    stiffness=0, damping=0, armature=0.03, friction=0.03,
                )
                for i in range(len(DOF_NAMES))
            }

        @configclass
        class _SceneCfg(InteractiveSceneCfg):
            env_spacing: float = 0.0
            ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane",
                                  spawn=sim_utils.GroundPlaneCfg())
            light = AssetBaseCfg(prim_path="/World/Light",
                                 spawn=sim_utils.DomeLightCfg(intensity=3000.0))
            robot: ArticulationCfg = BIO_ACT_CFG.replace(
                prim_path="/World/envs/env_.*/Robot", actuators=build_actuators()
            )

        sim_cfg = sim_utils.SimulationCfg(device=device, dt=dt)
        self.sim = SimulationContext(sim_cfg)
        self.scene = InteractiveScene(_SceneCfg(num_envs=1))
        self.sim.reset()
        self.robot: Articulation = self.scene["robot"]

        sim_bodies = list(self.robot.data.body_names)
        self.sim_joints = list(self.robot.data.joint_names)
        self.body_to_common = torch.tensor([sim_bodies.index(n) for n in BODY_NAMES],
                                           dtype=torch.long, device=device)
        self.dof_to_common = torch.tensor([self.sim_joints.index(n) for n in DOF_NAMES],
                                          dtype=torch.long, device=device)
        self.dof_to_sim = torch.tensor([DOF_NAMES.index(n) for n in self.sim_joints],
                                       dtype=torch.long, device=device)

        kp_common, kd_common = build_pd_gains()
        self.kp_common = kp_common.to(device)
        self.kd_common = kd_common.to(device)

        self.ctl = MuscleController(muscle_xml_path=MUSCLE_XML, rig_path=BIO_XML,
                                    device=torch.device(device))
        self.ctl.prepare(BODY_NAMES, DOF_NAMES)
        self.n_muscles = self.ctl.n_muscles
        if use_scale_map:
            self.ctl.set_muscle_scales(MUSCLE_SCALE_MAP, max_scale=20.0)

    def _prepare_motion(self, motion, max_frames):
        dof_pos_ref = local_rotation_to_dof(motion.local_rotation).to(torch.float)
        root_pos_ref = motion.root_translation
        root_rot_ref = motion.global_rotation[:, 0]
        T = min(dof_pos_ref.shape[0], max_frames)
        root_vel = torch.zeros_like(root_pos_ref)
        if T > 1:
            root_vel[1:] = (root_pos_ref[1:] - root_pos_ref[:-1]) * float(motion.fps)
        substeps = max(1, round(240.0 / float(motion.fps)))
        return dof_pos_ref, root_pos_ref, root_rot_ref, root_vel, T, substeps

    def track(self, motion_file=None, motion=None, max_frames=120, method="ls",
              max_iter=5, kp_scale=1.0, muscle_scale=1.0, pd_only=False):
        """跟踪一段 motion, 返回指标与轨迹 dict。

        Returns:
            dict: tracking_err (rad), torque_match, act_mean, act_sat, nan_flag,
                  q (T,50) common 顺序, q_ref, a (T,284), tau_des, tau_muscle。
        """
        if motion is None:
            assert motion_file is not None, "需提供 motion_file 或 motion"
            motion = SkeletonMotion.from_file(motion_file)
        dof_pos_ref, root_pos_ref, root_rot_ref, root_vel, T, substeps = \
            self._prepare_motion(motion, max_frames)

        kp = (self.kp_common * kp_scale).unsqueeze(0)
        kd = (self.kd_common * kp_scale).unsqueeze(0)
        device = self.device

        # 重置到首帧
        q0_sim = dof_pos_ref[0].to(device)[self.dof_to_sim].unsqueeze(0)
        self.robot.write_joint_state_to_sim(q0_sim, torch.zeros_like(q0_sim))
        init_root = torch.cat([
            root_pos_ref[0].to(device).unsqueeze(0),
            rotations.xyzw_to_wxyz(root_rot_ref[0].to(device).unsqueeze(0)),
            root_vel[0].to(device).unsqueeze(0),
            torch.zeros(1, 3, device=device),
        ], dim=-1)
        self.robot.write_root_state_to_sim(init_root)

        log = {"q": [], "q_ref": [], "a": [], "tau_des": [], "tau_muscle": [], "body_pos": []}
        last_act = None

        for t in range(T):
            root_state = torch.cat([
                root_pos_ref[t].to(device).unsqueeze(0),
                rotations.xyzw_to_wxyz(root_rot_ref[t].to(device).unsqueeze(0)),
                root_vel[t].to(device).unsqueeze(0),
                torch.zeros(1, 3, device=device),
            ], dim=-1)
            self.robot.write_root_state_to_sim(root_state)
            q_ref_common = dof_pos_ref[t].to(device)

            for _ in range(substeps):
                body_pos = self.robot.data.body_pos_w.clone()[:, self.body_to_common, :]
                body_quat_xyzw = wxyz_to_xyzw(self.robot.data.body_quat_w.clone())[:, self.body_to_common, :]
                com = self.robot.data.body_com_pos_w.clone()[:, self.body_to_common, :]

                jac = self.robot.root_physx_view.get_jacobians()
                if jac.shape[-1] != len(DOF_NAMES):
                    if jac.shape[-1] > len(DOF_NAMES):
                        jac = jac[..., -len(DOF_NAMES):]
                    else:
                        raise RuntimeError(f"jacobian DOF 维度过小: {jac.shape}")
                jac = jac[:, self.body_to_common, :, :]
                jac = jac[:, :, :, self.dof_to_common]

                q_sim = self.robot.data.joint_pos
                qd_sim = self.robot.data.joint_vel
                q_common = q_sim[:, self.dof_to_common]
                qd_common = qd_sim[:, self.dof_to_common]
                tau_des = kp * (q_ref_common.unsqueeze(0) - q_common) - kd * qd_common
                tau_des = torch.clip(tau_des, -self.torque_limit, self.torque_limit)

                if pd_only:
                    a = torch.zeros(1, self.n_muscles, device=device)
                    tau_muscle = tau_des
                else:
                    JtA, b = self.ctl.update_muscle_features(body_pos, body_quat_xyzw, com, jac)
                    if muscle_scale != 1.0:
                        JtA = JtA * muscle_scale
                        b = b * muscle_scale
                    a, tau_muscle = optimize_act(JtA, b, tau_des, method=method,
                                                 max_iter=max_iter, last_act=last_act)
                    last_act = a.detach()

                tau_muscle_sim = tau_muscle[:, self.dof_to_sim]
                self.robot.set_joint_effort_target(tau_muscle_sim)
                self.scene.write_data_to_sim()
                self.sim.step()
                self.scene.update(self.dt)

            log["q"].append(self.robot.data.joint_pos[0].detach().cpu().numpy())
            log["q_ref"].append(q_ref_common.cpu().numpy())
            log["a"].append(a[0].detach().cpu().numpy())
            log["tau_des"].append(tau_des[0].detach().cpu().numpy())
            log["tau_muscle"].append(tau_muscle[0].detach().cpu().numpy())
            log["body_pos"].append(
                self.robot.data.body_pos_w.clone()[:, self.body_to_common, :][0].detach().cpu().numpy()
            )

        dof_to_common_cpu = self.dof_to_common.cpu().numpy()
        q = np.stack(log["q"])[:, dof_to_common_cpu]  # sim -> common
        q_ref = np.stack(log["q_ref"])
        a_all = np.stack(log["a"])
        tau_des_all = np.stack(log["tau_des"])
        tau_mus_all = np.stack(log["tau_muscle"])

        tracking_err = float(np.abs(q - q_ref).mean())
        denom = np.linalg.norm(tau_des_all, axis=-1) + 1e-6
        torque_match = float(np.mean(np.linalg.norm(tau_mus_all - tau_des_all, axis=-1) / denom))
        act_mean = float(a_all.mean())
        act_sat = float(((a_all < 0.02) | (a_all > 0.98)).mean())
        nan_flag = bool(np.isnan(q).any() or np.isinf(q).any())

        return dict(tracking_err=tracking_err, torque_match=torque_match,
                    act_mean=act_mean, act_sat=act_sat, nan_flag=nan_flag,
                    q=q, q_ref=q_ref, a=a_all, tau_des=tau_des_all, tau_muscle=tau_mus_all,
                    body_pos=np.stack(log["body_pos"]))

    def kinematic_reference(self, motion_file=None, motion=None, max_frames=120):
        """返回参考姿态的刚体世界位置 (T, 23, 3) —— 根节点对齐 + 关节设为 q_ref, 不做物理步进。

        用于可视化时叠加"参考骨架"(红线), 与肌肉驱动实际轨迹(蓝线)对比。
        """
        if motion is None:
            assert motion_file is not None
            motion = SkeletonMotion.from_file(motion_file)
        dof_pos_ref, root_pos_ref, root_rot_ref, root_vel, T, _ = \
            self._prepare_motion(motion, max_frames)

        body_pos_list = []
        for t in range(T):
            root_state = torch.cat([
                root_pos_ref[t].to(self.device).unsqueeze(0),
                rotations.xyzw_to_wxyz(root_rot_ref[t].to(self.device).unsqueeze(0)),
                root_vel[t].to(self.device).unsqueeze(0),
                torch.zeros(1, 3, device=self.device),
            ], dim=-1)
            self.robot.write_root_state_to_sim(root_state)
            q_sim = dof_pos_ref[t].to(self.device)[self.dof_to_sim].unsqueeze(0)
            self.robot.write_joint_state_to_sim(q_sim, torch.zeros_like(q_sim))
            self.scene.write_data_to_sim()
            body_pos = self.robot.data.body_pos_w.clone()[:, self.body_to_common, :][0]
            body_pos_list.append(body_pos.detach().cpu().numpy())
        return np.stack(body_pos_list)  # (T, 23, 3)

    def close(self):
        """释放引用。注意: 不调用 simulation_app.close() —— Isaac Sim 关闭常会挂起;
        调用方在脚本末尾直接用 os._exit(0) 结束进程即可。"""
        self._closed = True
