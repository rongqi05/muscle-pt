"""MuJoCo 版直接肌肉控制 (替代 Isaac Lab 的刚体仿真, 无需 GPU)。

技术路线不变:
    BVH/BIO q_ref -> PD desired torque -> optimize 284 activation
        -> Hill muscle -> 50 torque -> MuJoCo 刚体仿真

保留自己的实现:
  - Hill 肌肉模型: protomotions/utils/muscle_control.py (MuscleController)
  - 284 激活 + recruitment optimizer: 本模块从 direct_muscle 复用 optimize_act
  - q_ref 生成: direct_muscle.local_rotation_to_dof

仅替换刚体仿真: Isaac Lab (GPU) -> MuJoCo (CPU, mujoco 3.11)。

用法:
    from protomotions.utils.direct_muscle_mujoco import DirectMuscleTrackerMujoco
    tracker = DirectMuscleTrackerMujoco(use_scale_map=True)
    res = tracker.track("data/cmu_bio_npy/009/09_12.npy", max_frames=120)
    print(res["tracking_err"])
"""

import os
import numpy as np
import torch
import mujoco

from poselib.skeleton.skeleton3d import SkeletonMotion
from protomotions.utils.muscle_control import MuscleController
from protomotions.utils.direct_muscle import (
    BODY_NAMES,
    DOF_NAMES,
    MUSCLE_SCALE_MAP,
    MUSCLE_XML,
    build_pd_gains,
    local_rotation_to_dof,
    optimize_act,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIO_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "mjcf", "bio.xml")
BIO_FLOOR_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "mjcf", "bio_floor.xml")


def _mat_to_quat_xyzw(xmat: np.ndarray) -> np.ndarray:
    """(N,3,3) 旋转矩阵 -> (N,4) 四元数 xyzw (w 在最后)。"""
    N = xmat.shape[0]
    out = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        mujoco.mju_mat2Quat(out[i], xmat[i].reshape(9))  # wxyz
    return out[:, [1, 2, 3, 0]]  # wxyz -> xyzw


class DirectMuscleTrackerMujoco:
    """MuJoCo 版直接肌肉控制跟踪器 (CPU)。

    与 Isaac Lab 版 DirectMuscleTracker 同接口风格, 但刚体仿真在 MuJoCo 上。
    """

    def __init__(self, use_scale_map: bool = True, torque_limit: float = 1000.0,
                 timestep: float = 1.0 / 240.0):
        self.model = mujoco.MjModel.from_xml_path(BIO_FLOOR_XML)
        # 关掉 bio.xml 关节自带的弹簧/阻尼 (Isaac Lab 侧执行器是 stiffness=0, damping=0)
        self.model.jnt_stiffness[:] = 0.0
        self.model.dof_damping[:] = 0.0
        # 关键: 加上 armature/friction (与 Isaac Lab IdealPDActuator armature=0.03,
        # friction=0.03 一致)。缺少 armature 时高刚度 PD 会让自由关节发散 (QACC 爆炸)。
        self.model.dof_armature[:] = 0.03
        self.model.dof_frictionloss[:] = 0.03
        self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.torque_limit = torque_limit

        # 名字 -> MuJoCo id 映射
        self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                         for n in BODY_NAMES]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                          for n in DOF_NAMES]
        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids]  # (50,)
        self.dof_adr = self.model.jnt_dofadr[self.joint_ids]    # (50,)
        self.n_act = len(DOF_NAMES)                             # 50
        self.dof_act_start = self.model.nv - self.n_act          # 6 (自由关节 6 DOF)

        # PD 增益 (common 顺序)
        kp, kd = build_pd_gains()
        self.kp = kp  # (50,)
        self.kd = kd

        # 肌肉模型 (复用自己的 Hill 肌肉, 全部 CPU)
        self.ctl = MuscleController(muscle_xml_path=MUSCLE_XML, rig_path=BIO_XML,
                                    device=torch.device("cpu"))
        self.ctl.prepare(BODY_NAMES, DOF_NAMES)
        self.n_muscles = self.ctl.n_muscles
        if use_scale_map:
            self.ctl.set_muscle_scales(MUSCLE_SCALE_MAP, max_scale=20.0)

        # 雅可比缓冲
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    # ------------------------------------------------------------------
    def set_hemiplegia_strength(self, side: str = "L", strength: float = 1.0) -> int:
        """偏瘫肌力减弱: 患侧 (side='L'/'R') 所有肌肉 f0 乘 strength (如 0.8/0.6/0.4)。

        在 scale_map 补偿之后原地乘, 即患侧整体强度降为健侧的 strength 倍。
        返回患侧肌肉数量。
        """
        prefix = str(side).upper() + "_"
        scales = {m.name: float(strength)
                  for m in self.ctl.muscle_char.muscles if m.name.startswith(prefix)}
        self.ctl.set_muscle_scales(scales, max_scale=100.0)
        self.affected_side = prefix[0]
        return len(scales)

    def _set_root(self, root_pos, root_rot_xyzw, root_vel):
        """传送自由关节 (根) 到给定位姿/速度。root_rot_xyzw: (4,) xyzw。"""
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_rot_xyzw[[3, 0, 1, 2]]  # xyzw -> wxyz
        self.data.qvel[0:3] = root_vel
        self.data.qvel[3:6] = 0.0

    def _read_state(self):
        q = torch.from_numpy(self.data.qpos[self.qpos_adr].copy()).float()   # (50,)
        qd = torch.from_numpy(self.data.qvel[self.dof_adr].copy()).float()   # (50,)
        return q, qd

    def _body_states_and_jac(self):
        """返回 (body_pos, body_rot_xyzw, com, jac), 均在 BODY_NAMES 顺序。"""
        body_pos = self.data.xpos[self.body_ids]                     # (23,3)
        body_rot = _mat_to_quat_xyzw(self.data.xmat[self.body_ids])  # (23,4) xyzw
        com = self.data.xipos[self.body_ids]                         # (23,3)
        jac = np.zeros((23, 6, self.n_act), dtype=np.float64)
        for i, bid in enumerate(self.body_ids):
            mujoco.mj_jacBodyCom(self.model, self.data, self._jacp, self._jacr, bid)
            jac[i, 0:3, :] = self._jacp[:, self.dof_act_start:]
            jac[i, 3:6, :] = self._jacr[:, self.dof_act_start:]
        return body_pos, body_rot, com, jac

    # ------------------------------------------------------------------
    def track(self, motion_file=None, motion=None, max_frames=120, method="lbfgs",
              max_iter=50, kp_scale=1.0, muscle_scale=1.0, pd_only=False):
        if motion is None:
            motion = SkeletonMotion.from_file(motion_file)

        dof_pos_ref = local_rotation_to_dof(motion.local_rotation).float()  # (T,50)
        root_pos_ref = motion.root_translation
        root_rot_ref = motion.global_rotation[:, 0]  # (T,4) xyzw
        T = min(dof_pos_ref.shape[0], max_frames)
        root_vel = torch.zeros_like(root_pos_ref)
        if T > 1:
            root_vel[1:] = (root_pos_ref[1:] - root_pos_ref[:-1]) * float(motion.fps)
        substeps = max(1, round(int(1.0 / self.model.opt.timestep) / float(motion.fps)))

        kp = self.kp * kp_scale
        kd = self.kd * kp_scale

        # 初始状态: 根对齐首帧, 关节对齐首帧 q_ref
        self._set_root(root_pos_ref[0].numpy(), root_rot_ref[0].numpy(), root_vel[0].numpy())
        self.data.qpos[self.qpos_adr] = dof_pos_ref[0].numpy()
        self.data.qvel[self.dof_adr] = 0.0
        self.data.ctrl[:] = 0.0  # 禁用 bio.xml 的 motor 执行器
        mujoco.mj_forward(self.model, self.data)

        log = {"q": [], "q_ref": [], "a": [], "tau_des": [], "tau_muscle": [], "body_pos": []}
        last_act = None

        for t in range(T):
            # 每帧传送根节点一次 (与 Isaac Lab 版一致)
            self._set_root(root_pos_ref[t].numpy(), root_rot_ref[t].numpy(), root_vel[t].numpy())
            q_ref = dof_pos_ref[t]

            for _ in range(substeps):
                mujoco.mj_forward(self.model, self.data)
                q, qd = self._read_state()

                tau_des = kp * (q_ref - q) - kd * qd
                tau_des = torch.clip(tau_des, -self.torque_limit, self.torque_limit)

                if pd_only:
                    a = torch.zeros(self.n_muscles)
                    tau_muscle = tau_des
                else:
                    body_pos, body_rot, com, jac = self._body_states_and_jac()
                    JtA, b = self.ctl.update_muscle_features(
                        torch.from_numpy(body_pos)[None].float(),
                        torch.from_numpy(body_rot)[None].float(),
                        torch.from_numpy(com)[None].float(),
                        torch.from_numpy(jac)[None].float(),
                    )
                    if muscle_scale != 1.0:
                        JtA = JtA * muscle_scale
                        b = b * muscle_scale
                    a, tau_muscle = optimize_act(JtA, b, tau_des[None], method=method,
                                                 max_iter=max_iter, last_act=last_act)
                    a = a[0]
                    tau_muscle = tau_muscle[0]
                    last_act = a.detach()

                # 施加肌肉力矩到 MuJoCo 关节
                self.data.qfrc_applied[self.dof_adr] = tau_muscle.numpy()
                self.data.ctrl[:] = 0.0
                mujoco.mj_step(self.model, self.data)

            log["q"].append(self.data.qpos[self.qpos_adr].copy())
            log["q_ref"].append(q_ref.numpy())
            log["a"].append(a.numpy())
            log["tau_des"].append(tau_des.numpy())
            log["tau_muscle"].append(tau_muscle.numpy())
            log["body_pos"].append(self.data.xpos[self.body_ids].copy())

        q = np.stack(log["q"])
        q_ref = np.stack(log["q_ref"])
        a_all = np.stack(log["a"])
        tau_des_all = np.stack(log["tau_des"])
        tau_mus_all = np.stack(log["tau_muscle"])
        body_pos = np.stack(log["body_pos"])

        tracking_err = float(np.abs(q - q_ref).mean())
        denom = np.linalg.norm(tau_des_all, axis=-1) + 1e-6
        torque_match = float(np.mean(np.linalg.norm(tau_mus_all - tau_des_all, axis=-1) / denom))
        act_mean = float(a_all.mean())
        act_sat = float(((a_all < 0.02) | (a_all > 0.98)).mean())
        nan_flag = bool(np.isnan(q).any() or np.isinf(q).any())

        return dict(tracking_err=tracking_err, torque_match=torque_match,
                    act_mean=act_mean, act_sat=act_sat, nan_flag=nan_flag,
                    q=q, q_ref=q_ref, a=a_all, tau_des=tau_des_all,
                    tau_muscle=tau_mus_all, body_pos=body_pos)

    def kinematic_reference(self, motion_file=None, motion=None, max_frames=120):
        """参考姿态的刚体世界位置 (T,23,3): 根对齐 + 关节设为 q_ref, 不做物理步进。"""
        if motion is None:
            motion = SkeletonMotion.from_file(motion_file)
        dof_pos_ref = local_rotation_to_dof(motion.local_rotation).float()
        root_pos_ref = motion.root_translation
        root_rot_ref = motion.global_rotation[:, 0]
        T = min(dof_pos_ref.shape[0], max_frames)
        root_vel = torch.zeros_like(root_pos_ref)
        if T > 1:
            root_vel[1:] = (root_pos_ref[1:] - root_pos_ref[:-1]) * float(motion.fps)

        ref = []
        for t in range(T):
            self._set_root(root_pos_ref[t].numpy(), root_rot_ref[t].numpy(), root_vel[t].numpy())
            self.data.qpos[self.qpos_adr] = dof_pos_ref[t].numpy()
            mujoco.mj_forward(self.model, self.data)
            ref.append(self.data.xpos[self.body_ids].copy())
        return np.stack(ref)

    # ------------------------------------------------------------------
    # 肌肉可视化
    # ------------------------------------------------------------------
    def compute_muscle_points(self):
        """返回 (M, maxW, 3) 肌肉 waypoint 世界坐标 + (M,) 每块肌肉的有效 waypoint 数。

        复刻 MuscleController.update_muscle_features 里的 waypoint 世界坐标计算
        (p_world = Σ w·(xpos + R·local_pt)), 用于把肌肉路径渲染成线段。
        """
        c = self.ctl.muscle_char
        pos = torch.from_numpy(self.data.xpos[self.body_ids]).float()      # (23,3)
        R = torch.from_numpy(self.data.xmat[self.body_ids].reshape(-1, 3, 3)).float()  # (23,3,3)
        padded_idx = torch.as_tensor(c._padded_backend_idx, dtype=torch.long)      # (M,maxW,maxK)
        local_pts = torch.as_tensor(c._padded_local_pts, dtype=torch.float32)      # (M,maxW,maxK,3)
        weights = torch.as_tensor(c._padded_weights, dtype=torch.float32)          # (M,maxW,maxK)

        safe_idx = padded_idx.clamp(min=0)
        x_wp = pos[safe_idx]                                    # (M,maxW,maxK,3)
        R_wp = R[safe_idx]                                      # (M,maxW,maxK,3,3)
        p_world_infl = x_wp + torch.matmul(R_wp, local_pts.unsqueeze(-1)).squeeze(-1)  # (M,maxW,maxK,3)
        p_world = (p_world_infl * weights.unsqueeze(-1)).sum(dim=2)  # (M,maxW,3)
        n_waypoints = (weights.sum(dim=2) > 0).sum(dim=1).numpy()     # (M,)
        return p_world.numpy(), n_waypoints

    @staticmethod
    def _activation_color(a: float):
        """激活 -> RGBA: 深灰(0) -> 亮红(1), alpha 随激活增强 (医学灰红风格)。"""
        t = float(np.clip(a, 0.0, 1.0))
        rgb = np.array([0.30 + 0.70 * t,
                        0.30 * (1.0 - t) + 0.12 * t,
                        0.32 * (1.0 - t) + 0.06 * t])
        alpha = 0.45 + 0.55 * t
        return np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.float32)

    def render_video(self, motion_file=None, motion=None, output="output/mj_muscle.mp4",
                     max_frames=120, method="lbfgs", max_iter=50, kp_scale=1.0,
                     muscle_scale=1.0, fps=60, width=960, height=720,
                     show_muscles=True, muscle_radius=0.008, muscle_stride=1):
        """跟踪 motion 并用 MuJoCo 离屏渲染成视频 (骨骼 + 按激活上色的肌肉线段)。

        高帧率由物理子步插值实现: 每 mocap 帧内有 substeps 个物理子步 (240Hz),
        每 render_every 个子步渲染一帧, 动作由物理仿真自然插值 (比逐 mocap 帧更顺滑)。
        show_muscles=True 时把 284 条肌肉路径画成胶囊线段 (灰=未激活, 红=强收缩)。
        muscle_stride: 每隔几个 waypoint 取一个 (降采样, 加速渲染)。
        """
        if motion is None:
            motion = SkeletonMotion.from_file(motion_file)

        dof_pos_ref = local_rotation_to_dof(motion.local_rotation).float()
        root_pos_ref = motion.root_translation
        root_rot_ref = motion.global_rotation[:, 0]
        T = min(dof_pos_ref.shape[0], max_frames)
        root_vel = torch.zeros_like(root_pos_ref)
        if T > 1:
            root_vel[1:] = (root_pos_ref[1:] - root_pos_ref[:-1]) * float(motion.fps)
        substeps = max(1, round(int(1.0 / self.model.opt.timestep) / float(motion.fps)))

        kp = self.kp * kp_scale
        kd = self.kd * kp_scale

        renderer = mujoco.Renderer(self.model, height, width)  # 注意: Renderer(model, height, width)
        cam = mujoco.MjvCamera()
        cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
        opt = mujoco.MjvOption()

        # 骨骼视觉淡化 (半透明), 让肌肉线条更突出
        try:
            mid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "MatBone")
            if mid >= 0:
                self.model.mat_rgba[mid] = [0.85, 0.85, 0.88, 0.55]
        except Exception:
            pass

        # 初始状态
        self._set_root(root_pos_ref[0].numpy(), root_rot_ref[0].numpy(), root_vel[0].numpy())
        self.data.qpos[self.qpos_adr] = dof_pos_ref[0].numpy()
        self.data.qvel[self.dof_adr] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        frames = []
        last_act = None
        a = torch.zeros(self.n_muscles)
        # 每 mocap 帧渲染的子步间隔 (物理插值出高帧率): render_every = substeps*mocap_fps/fps
        render_every = max(1, round(substeps * float(motion.fps) / fps))
        n_rendered = 0
        for t in range(T):
            self._set_root(root_pos_ref[t].numpy(), root_rot_ref[t].numpy(), root_vel[t].numpy())
            q_ref = dof_pos_ref[t]
            for s in range(substeps):
                mujoco.mj_forward(self.model, self.data)
                q, qd = self._read_state()
                tau_des = torch.clip(kp * (q_ref - q) - kd * qd, -self.torque_limit, self.torque_limit)
                body_pos, body_rot, com, jac = self._body_states_and_jac()
                JtA, b = self.ctl.update_muscle_features(
                    torch.from_numpy(body_pos)[None].float(),
                    torch.from_numpy(body_rot)[None].float(),
                    torch.from_numpy(com)[None].float(),
                    torch.from_numpy(jac)[None].float())
                if muscle_scale != 1.0:
                    JtA = JtA * muscle_scale
                    b = b * muscle_scale
                a, tau_muscle = optimize_act(JtA, b, tau_des[None], method=method,
                                             max_iter=max_iter, last_act=last_act)
                a = a[0]; tau_muscle = tau_muscle[0]
                last_act = a.detach()
                self.data.qfrc_applied[self.dof_adr] = tau_muscle.numpy()
                self.data.ctrl[:] = 0.0
                mujoco.mj_step(self.model, self.data)

                # 在整数倍 render_every 的子步处渲染 (中间姿态由物理插值, 视频更顺滑)
                if (s + 1) % render_every != 0:
                    continue
                mujoco.mj_forward(self.model, self.data)
                pelvis = self.data.xpos[1]
                cam.lookat[:] = pelvis + np.array([0.0, 0.0, 0.45])
                cam.distance = 4.4
                cam.azimuth = -60.0
                cam.elevation = -14.0
                renderer.update_scene(self.data, camera=cam, scene_option=opt)
                if show_muscles:
                    p_world, n_wp = self.compute_muscle_points()
                    anp = a.numpy()
                    for m in range(p_world.shape[0]):
                        t = anp[m]
                        col = self._activation_color(t)
                        # 半径随激活: 未激活细弱 (30%), 强收缩粗壮 (100%)
                        rad = max(0.0015, muscle_radius * (0.30 + 0.70 * t))
                        W = int(n_wp[m])
                        for w in range(0, W - 1, max(1, muscle_stride)):
                            g = renderer.scene.geoms[renderer.scene.ngeom]
                            mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                                                np.zeros(3), np.zeros(3), np.eye(3).flatten(), col)
                            mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                                                 rad,
                                                 p_world[m, w], p_world[m, min(w + 1, W - 1)])
                            renderer.scene.ngeom += 1
                frames.append(renderer.render().copy())
                n_rendered += 1
                if n_rendered % 60 == 0:
                    print(f"  rendered {n_rendered} frames")

        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        import imageio
        imageio.mimsave(output, frames, fps=fps)
        print(f"视频已保存: {output} ({len(frames)} 帧 @ {fps}fps)")
        return output
