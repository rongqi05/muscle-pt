#!/usr/bin/env python3
"""用 MuJoCo 官方 viewer 实时交互查看肌肉驱动行走 demo。

独立于现有代码 (只 import 复用 DirectMuscleTrackerMujoco, 不修改任何现有文件)。
无第三方依赖: 使用 MuJoCo 自带的 mujoco.viewer (需要 glfw, 已在 env_isaaclab 中)。

用法 (仓库根目录):
    PYTHONPATH=. python mujoco_demo/viewer_demo/view_demo.py \
        --motion data/cmu_bio_npy/009/09_12.npy --max-frames 480 --loop

窗口交互 (官方 viewer):
    鼠标左键拖拽=旋转  右键拖拽=平移  滚轮=缩放  关闭窗口=退出
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from poselib.skeleton.skeleton3d import SkeletonMotion  # noqa: E402
from protomotions.utils.direct_muscle import local_rotation_to_dof, optimize_act  # noqa: E402
from protomotions.utils.direct_muscle_mujoco import DirectMuscleTrackerMujoco  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", default="data/cmu_bio_npy/009/09_12.npy")
    parser.add_argument("--max-frames", type=int, default=480,
                        help="播放的 mocap 帧数 (480=整段 16 秒)")
    parser.add_argument("--max-iter", type=int, default=50, help="激活优化迭代 (越小越快)")
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--render-every", type=int, default=2,
                        help="每几个物理子步刷新一次窗口 (1=最流畅/最慢)")
    parser.add_argument("--loop", action="store_true", help="循环播放直到关闭窗口")
    args = parser.parse_args()

    import mujoco
    from mujoco import viewer as mj_viewer

    tracker = DirectMuscleTrackerMujoco(use_scale_map=True)
    model, data = tracker.model, tracker.data

    handle = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    print("窗口已打开。鼠标左键旋转 / 右键平移 / 滚轮缩放 / 关闭窗口退出。")

    motion = SkeletonMotion.from_file(args.motion)
    dof_pos_ref = local_rotation_to_dof(motion.local_rotation).float()
    root_pos_ref = motion.root_translation
    root_rot_ref = motion.global_rotation[:, 0]
    T = min(dof_pos_ref.shape[0], args.max_frames)
    root_vel = torch.zeros_like(root_pos_ref)
    if T > 1:
        root_vel[1:] = (root_pos_ref[1:] - root_pos_ref[:-1]) * float(motion.fps)
    substeps = max(1, round(int(1.0 / model.opt.timestep) / float(motion.fps)))

    kp = tracker.kp * args.kp_scale
    kd = tracker.kd * args.kp_scale

    # 初始状态对齐首帧
    tracker._set_root(root_pos_ref[0].numpy(), root_rot_ref[0].numpy(), root_vel[0].numpy())
    data.qpos[tracker.qpos_adr] = dof_pos_ref[0].numpy()
    data.qvel[tracker.dof_adr] = 0.0
    data.ctrl[:] = 0.0
    import mujoco
    mujoco.mj_forward(model, data)

    last_act = None
    a = torch.zeros(tracker.n_muscles)
    step = 0
    try:
        while handle.is_running():
            t = step // substeps
            if t >= T:
                if not args.loop:
                    break
                step = 0
                t = 0
                last_act = None
                tracker._set_root(root_pos_ref[0].numpy(), root_rot_ref[0].numpy(),
                                  root_vel[0].numpy())
                data.qpos[tracker.qpos_adr] = dof_pos_ref[0].numpy()
                data.qvel[tracker.dof_adr] = 0.0

            if step % substeps == 0:
                tracker._set_root(root_pos_ref[t].numpy(), root_rot_ref[t].numpy(),
                                  root_vel[t].numpy())
            q_ref = dof_pos_ref[t]

            mujoco.mj_forward(model, data)
            q, qd = tracker._read_state()
            tau_des = torch.clip(kp * (q_ref - q) - kd * qd,
                                 -tracker.torque_limit, tracker.torque_limit)
            body_pos, body_rot, com, jac = tracker._body_states_and_jac()
            JtA, b = tracker.ctl.update_muscle_features(
                torch.from_numpy(body_pos)[None].float(),
                torch.from_numpy(body_rot)[None].float(),
                torch.from_numpy(com)[None].float(),
                torch.from_numpy(jac)[None].float())
            a, tau_muscle = optimize_act(JtA, b, tau_des[None],
                                         method="lbfgs",
                                         max_iter=args.max_iter,
                                         last_act=last_act)
            a = a[0]; tau_muscle = tau_muscle[0]
            last_act = a.detach()
            data.qfrc_applied[tracker.dof_adr] = tau_muscle.numpy()
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)

            if step % args.render_every == 0:
                handle.sync()
            step += 1
    finally:
        # 官方 viewer 窗口在后台线程运行, 进程退出时其 glfw 清理会段错误;
        # 用 os._exit 直接退出 (窗口随进程关闭), 不再走解释器清理
        pass
    print("已退出 viewer。")
    os._exit(0)


if __name__ == "__main__":
    main()
