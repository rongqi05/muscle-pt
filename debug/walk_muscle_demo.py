"""Level 7: 肌肉激活控制完整技术路线(无 RL, 直接逐帧优化)。

技术路线:
    BVH → BIO q_ref → PD desired torque → optimization → 284 activation
        → muscle model (Hill) → 50 torque → Isaac Lab

流程:
  1. 加载重定向后的 .npy(SkeletonMotion, BIO 骨架)→ q_ref (50 维 dof_pos)
  2. 在 Isaac Lab 中 spawn bio humanoid(力矩驱动, stiffness=0)
  3. 构建 MuscleController(muscle284.xml + bio.xml), prepare 映射
  4. 每帧:
     - 根节点同步到参考姿态
     - 子步内: q/qd → PD 期望力矩 tau_des = kp*(q_ref-q) - kd*qd
     - 取刚体状态 + Jacobian → JtA/b
     - optimize_act: 解 284 个激活 a, 使 tau_muscle = a·JtA + b 逼近 tau_des
     - 将肌肉力矩写回 Isaac Lab(set_joint_effort_target)
  5. 记录 q/q_ref/tau_des/tau_muscle/activation, 输出指标

运行(仓库根目录):
    PYTHONPATH=. python debug/walk_muscle_demo.py \
        --motion data/cmu_bio_npy/008/xxx.npy --method ls --max-frames 120

参考: debug/walk_pd_demo.py (Level 6, 纯 PD) 与 debug/debug_jta.py (Level 4, JtA 验证)
"""

import argparse
import glob
import os
import sys

# 在导入 isaacsim 前设置显存优化, 减少肌肉 Jacobian OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

from isaac_utils import rotations, torch_utils  # noqa: E402
from isaac_utils.rotations import wxyz_to_xyzw  # noqa: E402
from poselib.skeleton.skeleton3d import SkeletonMotion  # noqa: E402
from protomotions.simulator.isaaclab.utils.robots import BIO_ACT_CFG  # noqa: E402
from protomotions.utils.muscle_control import MuscleController  # noqa: E402
from protomotions.utils.direct_muscle import MUSCLE_SCALE_MAP  # noqa: E402

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

# dof_body_ids / joint_axis 与 bio_act.yaml 一致 (来自 walk_pd_demo.py)
DOF_BODY_IDS = list(range(1, 23))
JOINT_AXIS = ['xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'xyz', 'y', 'xyz', 'xyz', 'xyz',
              'y', 'xyz', 'xyz', 'x', 'xyz', 'x', 'x', 'xyz', 'x', 'xyz', 'x', 'x']
_off = 0
DOF_OFFSETS = [0]
for ax in JOINT_AXIS:
    _off += len(ax)
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

    注意: 3 自由度关节用 quat_to_exp_map (角度在 [-pi, pi]), 而非 get_euler_xyz
    (后者返回 % 2pi, 会把小负角包成 ~6.28 rad, 越限后物理 NaN)。
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
class WalkSceneCfg(InteractiveSceneCfg):
    env_spacing: float = 0.0
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot: ArticulationCfg = BIO_ACT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot", actuators=build_actuators()
    )


def optimize_act(JtA, b, tau, method="ls", max_iter=5, last_act=None):
    """解 284 激活 a, 使 tau_pred = a·JtA + b 逼近 tau。

    与 protomotions/simulator/isaaclab/simulator.py::optimize_act 同款数学,
    此处内联以避免重依赖导入。method: 'ls' (最小二乘初值 + LBFGS 精修) 或 'lbfgs'。
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=str, required=True, help=".npy SkeletonMotion 文件(或通配)")
    parser.add_argument("--max-frames", type=int, default=120, help="最大帧数")
    parser.add_argument("--method", type=str, default="ls", choices=["ls", "lbfgs"],
                        help="激活优化方法")
    parser.add_argument("--max-iter", type=int, default=5, help="LBFGS 精修迭代次数")
    parser.add_argument("--kp-scale", type=float, default=1.0, help="PD 增益缩放 (越小期望力矩越软)")
    parser.add_argument("--muscle-scale", type=float, default=1.0, help="肌肉强度全局缩放 (f0)")
    parser.add_argument("--scale-map", action="store_true", help="应用偏弱肌肉的 f0 补偿倍率 (MUSCLE_SCALE_MAP)")
    parser.add_argument("--pd-only", action="store_true", help="跳过肌肉, 直接施加 PD 力矩 (对照基线)")
    parser.add_argument("--sweep", action="store_true", help="扫描 (kp_scale, muscle_scale) 网格")
    parser.add_argument("--batch", action="store_true", help="批量跑 --motion 通配的所有段, 输出统计")
    parser.add_argument("--fail-thresh", type=float, default=0.15, help="批量模式判定失败的角度阈值 (rad)")
    args = parser.parse_args()

    files = sorted(glob.glob(args.motion))
    assert files, f"未找到 motion 文件: {args.motion}"

    # ---- spawn 一次, 各段复用 ----
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim = SimulationContext(sim_utils.SimulationCfg(device=device, dt=1.0 / 240.0))
    scene = InteractiveScene(WalkSceneCfg(num_envs=1))
    sim.reset()
    robot: Articulation = scene["robot"]
    dt = 1.0 / 240.0

    sim_bodies = list(robot.data.body_names)
    sim_joints = list(robot.data.joint_names)
    # sim -> common 重排
    body_to_common = torch.tensor([sim_bodies.index(n) for n in BODY_NAMES], dtype=torch.long, device=device)
    dof_to_common = torch.tensor([sim_joints.index(n) for n in DOF_NAMES], dtype=torch.long, device=device)
    # common -> sim 重排
    dof_to_sim = torch.tensor([DOF_NAMES.index(n) for n in sim_joints], dtype=torch.long, device=device)
    dof_to_common_cpu = dof_to_common.cpu().numpy()  # sim -> common 重排索引

    kp_common, kd_common = build_pd_gains()
    torque_limit = 1000.0  # bio_act.yaml dof_effort_limits (N·m)

    ctl = MuscleController(muscle_xml_path=MUSCLE_XML, rig_path=BIO_XML, device=torch.device(device))
    ctl.prepare(BODY_NAMES, DOF_NAMES)
    n_muscles = ctl.n_muscles
    if args.scale_map:
        ctl.set_muscle_scales(MUSCLE_SCALE_MAP, max_scale=20.0)
    print(f"muscles: {n_muscles}, 方法: {args.method}, scale_map={args.scale_map}")

    def prepare_motion(motion):
        """BVH/BIO motion -> (dof_pos_ref, root_pos_ref, root_rot_ref, root_vel, T, substeps)。"""
        dof_pos_ref = local_rotation_to_dof(motion.local_rotation).to(torch.float)  # (T,50) common
        root_pos_ref = motion.root_translation     # (T,3)
        root_rot_ref = motion.global_rotation[:, 0]  # (T,4) xyzw
        T = min(dof_pos_ref.shape[0], args.max_frames)
        root_vel = torch.zeros_like(root_pos_ref)
        if T > 1:
            root_vel[1:] = (root_pos_ref[1:] - root_pos_ref[:-1]) * float(motion.fps)
        substeps = max(1, round(240.0 / float(motion.fps)))  # 30fps -> 8 substeps
        return dof_pos_ref, root_pos_ref, root_rot_ref, root_vel, T, substeps

    def run_once(dof_pos_ref, root_pos_ref, root_rot_ref, root_vel, T, substeps,
                 kp_scale: float, muscle_scale: float):
        """在给定 motion 与 (kp_scale, muscle_scale) 下完整跑一遍, 返回指标与轨迹。"""
        kp = (kp_common * kp_scale).to(device).unsqueeze(0)  # (1,50)
        kd = (kd_common * kp_scale).to(device).unsqueeze(0)

        # 重置机器人到首帧状态
        q0_sim = dof_pos_ref[0].to(device)[dof_to_sim].unsqueeze(0)
        robot.write_joint_state_to_sim(q0_sim, torch.zeros_like(q0_sim))
        init_root = torch.cat([
            root_pos_ref[0].to(device).unsqueeze(0),
            rotations.xyzw_to_wxyz(root_rot_ref[0].to(device).unsqueeze(0)),
            root_vel[0].to(device).unsqueeze(0),
            torch.zeros(1, 3, device=device),
        ], dim=-1)
        robot.write_root_state_to_sim(init_root)

        log = {"q": [], "q_ref": [], "a": [], "tau_des": [], "tau_muscle": []}
        last_act = None

        for t in range(T):
            # 同步根节点(拖拽式跟踪, 与 walk_pd_demo 一致)
            root_state = torch.cat([
                root_pos_ref[t].to(device).unsqueeze(0),
                rotations.xyzw_to_wxyz(root_rot_ref[t].to(device).unsqueeze(0)),
                root_vel[t].to(device).unsqueeze(0),
                torch.zeros(1, 3, device=device),
            ], dim=-1)
            robot.write_root_state_to_sim(root_state)

            q_ref_common = dof_pos_ref[t].to(device)  # (50,)

            for _ in range(substeps):
                # --- 刚体状态 (sim -> common) ---
                body_pos = robot.data.body_pos_w.clone()[:, body_to_common, :]      # (1,23,3)
                body_quat_xyzw = wxyz_to_xyzw(robot.data.body_quat_w.clone())[:, body_to_common, :]  # (1,23,4) xyzw
                com = robot.data.body_com_pos_w.clone()[:, body_to_common, :]      # (1,23,3)

                # --- Jacobian (sim -> common) ---
                jac = robot.root_physx_view.get_jacobians()
                if jac.shape[-1] != len(DOF_NAMES):
                    if jac.shape[-1] > len(DOF_NAMES):
                        jac = jac[..., -len(DOF_NAMES):]
                    else:
                        raise RuntimeError(f"jacobian DOF 维度过小: {jac.shape}")
                jac = jac[:, body_to_common, :, :]
                jac = jac[:, :, :, dof_to_common]

                # --- q_ref -> PD desired torque ---
                q_sim = robot.data.joint_pos
                qd_sim = robot.data.joint_vel
                q_common = q_sim[:, dof_to_common]
                qd_common = qd_sim[:, dof_to_common]
                tau_des = kp * (q_ref_common.unsqueeze(0) - q_common) - kd * qd_common  # (1,50)
                # 裁剪到关节力矩上限, 与生产 _compute_tau_from_actions 一致 (bio_act: ±1000 N·m)
                tau_des = torch.clip(tau_des, -torque_limit, torque_limit)

                # --- PD torque -> 284 activation (optimization) ---
                if args.pd_only:
                    # 对照: 直接用 PD 力矩 (肌肉模型不参与)
                    a = torch.zeros(1, n_muscles, device=device)
                    tau_muscle = tau_des
                else:
                    JtA, b = ctl.update_muscle_features(body_pos, body_quat_xyzw, com, jac)
                    if muscle_scale != 1.0:
                        JtA = JtA * muscle_scale
                        b = b * muscle_scale
                    a, tau_muscle = optimize_act(JtA, b, tau_des, method=args.method,
                                                 max_iter=args.max_iter, last_act=last_act)
                    last_act = a.detach()

                # --- muscle torque -> Isaac Lab ---
                tau_muscle_sim = tau_muscle[:, dof_to_sim]
                robot.set_joint_effort_target(tau_muscle_sim)
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)

            log["q"].append(robot.data.joint_pos[0].detach().cpu().numpy())       # sim order
            log["q_ref"].append(q_ref_common.cpu().numpy())                        # common order
            log["a"].append(a[0].detach().cpu().numpy())                           # (284,)
            log["tau_des"].append(tau_des[0].detach().cpu().numpy())               # (50,)
            log["tau_muscle"].append(tau_muscle[0].detach().cpu().numpy())         # (50,)

        q = np.stack(log["q"])[:, dof_to_common_cpu]  # sim -> common
        q_ref = np.stack(log["q_ref"])
        a_all = np.stack(log["a"])
        tau_des_all = np.stack(log["tau_des"])
        tau_mus_all = np.stack(log["tau_muscle"])

        tracking_err = np.abs(q - q_ref).mean()
        denom = np.linalg.norm(tau_des_all, axis=-1) + 1e-6
        torque_match = np.mean(np.linalg.norm(tau_mus_all - tau_des_all, axis=-1) / denom)
        act_mean = a_all.mean()
        act_sat = ((a_all < 0.02) | (a_all > 0.98)).mean()
        nan_flag = bool(np.isnan(q).any() or np.isinf(q).any())
        return dict(tracking_err=tracking_err, torque_match=torque_match,
                    act_mean=act_mean, act_sat=act_sat, nan_flag=nan_flag,
                    q=q, q_ref=q_ref, a=a_all, tau_des=tau_des_all, tau_muscle=tau_mus_all)

    if args.batch:
        print("=" * 78)
        print(f"批量评估: {len(files)} 段 (method={args.method}, max_iter={args.max_iter}, "
              f"kp_scale={args.kp_scale}, muscle_scale={args.muscle_scale}, scale_map={args.scale_map})")
        print(f"{'#':>3} {'motion':<34} {'err(°)':>8} {'NaN':>4} {'判定':>6}")
        print("-" * 78)
        errs, fails = [], 0
        for i, f in enumerate(files):
            motion = SkeletonMotion.from_file(f)
            d = prepare_motion(motion)
            r = run_once(*d, args.kp_scale, args.muscle_scale)
            deg = np.degrees(r["tracking_err"])
            failed = r["nan_flag"] or r["tracking_err"] > args.fail_thresh
            if failed:
                fails += 1
            else:
                errs.append(r["tracking_err"])
            name = os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f)
            print(f"{i:>3} {name:<34} {deg:>8.2f} {str(r['nan_flag']):>4} {'FAIL' if failed else 'OK':>6}")
            sys.stdout.flush()
        print("-" * 78)
        ok = len(files) - fails
        mean_deg = np.degrees(np.mean(errs)) if errs else float("nan")
        print(f"成功: {ok}/{len(files)}  失败率: {fails/len(files):.1%}")
        print(f"成功段平均跟踪误差: {mean_deg:.2f}°")
    elif args.sweep:
        motion = SkeletonMotion.from_file(files[0])
        d = prepare_motion(motion)
        grid = [
            (1.0, 1.0), (0.5, 1.0), (0.25, 1.0),
            (1.0, 2.0), (0.5, 2.0), (0.25, 2.0),
            (1.0, 4.0), (0.5, 4.0), (0.25, 4.0),
        ]
        print("=" * 72)
        print(f"{'kp_scale':>9} {'muscle_scale':>13} {'track_err(rad)':>14} {'torq_match':>11} {'act_mean':>9} {'act_sat':>8}")
        print("-" * 72)
        results = []
        for kps, ms in grid:
            r = run_once(*d, kps, ms)
            results.append((kps, ms, r))
            print(f"{kps:>9.2f} {ms:>13.2f} {r['tracking_err']:>14.4f} {r['torque_match']:>11.4f} {r['act_mean']:>9.4f} {r['act_sat']:>8.4f}")
            sys.stdout.flush()
        best = min(results, key=lambda x: x[2]["tracking_err"])
        print("-" * 72)
        print(f"最优: kp_scale={best[0]}, muscle_scale={best[1]}, "
              f"track_err={best[2]['tracking_err']:.4f} rad")
    else:
        motion = SkeletonMotion.from_file(files[0])
        print(f"motion: {files[0]}, frames={motion.global_rotation.shape[0]}, fps={motion.fps}")
        d = prepare_motion(motion)
        r = run_once(*d, args.kp_scale, args.muscle_scale)
        print("=" * 72)
        print(f"Level 7 — 肌肉激活控制完整路线结果 "
              f"(frames={d[4]}, method={args.method}, kp_scale={args.kp_scale}, muscle_scale={args.muscle_scale})")
        print("=" * 72)
        print(f"mean |q - q_ref|        = {r['tracking_err']:.4f} rad")
        print(f"mean ||tau_m - tau_des||/||tau_des|| = {r['torque_match']:.4f}")
        print(f"activation mean        = {r['act_mean']:.4f}")
        print(f"activation 饱和占比     = {r['act_sat']:.4f}")

        out = os.path.join(REPO_ROOT, "output", "walk_muscle_traj.npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out,
                 q=r["q"], q_ref=r["q_ref"], a=r["a"],
                 tau_des=r["tau_des"], tau_muscle=r["tau_muscle"],
                 sim_joints=np.array(sim_joints), dof_names=np.array(DOF_NAMES))
        print(f"记录已保存 -> {out}")
        print("RESULT: PASS" if r["tracking_err"] < args.fail_thresh else "RESULT: WARN (跟踪误差偏大)")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
