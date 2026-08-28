#!/usr/bin/env python3
"""自绘交互 viewer: 实时肌肉驱动行走 + 284 肌肉激活可视化。

用 glfw 窗口 + MuJoCo 离屏 Renderer, 每帧叠加 284 条肌肉线
(灰=未激活 -> 红=强收缩, 半径随激活), 与离屏视频渲染完全同款。

独立于现有代码 (只 import 复用, 不修改任何现有文件)。
依赖: glfw + PyOpenGL (env_isaaclab 已带), mujoco。

用法 (仓库根目录):
    PYTHONPATH=. python mujoco_demo/viewer_demo/view_demo_muscles.py \
        --motion data/cmu_bio_npy/009/09_12.npy --max-frames 480 --loop

交互: 左键拖拽=旋转  右键拖拽=平移  滚轮=缩放  关闭窗口=退出
"""

import argparse
import os
import sys

import numpy as np
import torch
import mujoco
import glfw
from OpenGL import GL as gl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from poselib.skeleton.skeleton3d import SkeletonMotion  # noqa: E402
from protomotions.utils.direct_muscle import local_rotation_to_dof, optimize_act  # noqa: E402
from protomotions.utils.direct_muscle_mujoco import DirectMuscleTrackerMujoco  # noqa: E402


class GlfwViewer:
    """glfw 窗口 + 纹理绘制 MuJoCo Renderer 输出的帧。"""

    def __init__(self, model, width=1280, height=960, title="Muscle walking"):
        self.model = model
        self.width, self.height = width, height
        if not glfw.init():
            raise RuntimeError("glfw init failed")
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            raise RuntimeError("window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        gl.glViewport(0, 0, width, height)

        # 全屏纹理 (显示渲染帧)
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        # 相机 (自由相机, 初始全身视角)
        self.cam = mujoco.MjvCamera()
        self.cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
        self.cam.azimuth = -60.0
        self.cam.elevation = -14.0
        self.cam.distance = 4.4
        self.lookat_offset = np.array([0.0, 0.0, 0.45])  # 相对骨盆
        self.last_x, self.last_y = None, None
        glfw.set_scroll_callback(self.window, self._scroll_cb)

    def _scroll_cb(self, window, xoff, yoff):
        self.cam.distance = float(np.clip(self.cam.distance * (1.1 ** (-yoff)),
                                          1.2, 15.0))

    def update_camera(self, pelvis):
        x, y = glfw.get_cursor_pos(self.window)
        if self.last_x is None:
            self.last_x, self.last_y = x, y
        dx, dy = x - self.last_x, y - self.last_y
        self.last_x, self.last_y = x, y

        lb = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT)
        rb = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT)
        if lb == glfw.PRESS:
            self.cam.azimuth -= dx * 0.35
            self.cam.elevation = float(np.clip(self.cam.elevation - dy * 0.35,
                                               -85.0, 85.0))
        if rb == glfw.PRESS:
            # 沿相机右/上轴平移 lookat
            az = np.radians(self.cam.azimuth)
            el = np.radians(self.cam.elevation)
            fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                            np.sin(el)])
            right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
            right /= np.linalg.norm(right) + 1e-9
            up = np.cross(right, fwd)
            scale = self.cam.distance * 0.0012
            self.lookat_offset = self.lookat_offset + (-dx * right + dy * up) * scale
        self.cam.lookat[:] = pelvis + self.lookat_offset

    def render(self, img):
        """img: (H, W, 3) uint8。渲染前恢复 glfw 窗口上下文
        (mujoco Renderer.render 可能切换了 OpenGL 上下文)。"""
        glfw.make_context_current(self.window)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                           gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-1, 1, -1, 1, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glColor3f(1, 1, 1)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 1); gl.glVertex2f(-1, -1)
        gl.glTexCoord2f(1, 1); gl.glVertex2f(1, -1)
        gl.glTexCoord2f(1, 0); gl.glVertex2f(1, 1)
        gl.glTexCoord2f(0, 0); gl.glVertex2f(-1, 1)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def should_close(self):
        return glfw.window_should_close(self.window)

    def close(self):
        glfw.destroy_window(self.window)
        glfw.terminate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", default="data/cmu_bio_npy/009/09_12.npy")
    parser.add_argument("--max-frames", type=int, default=480)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--render-every", type=int, default=2,
                        help="每几个物理子步渲染一帧")
    parser.add_argument("--muscle-stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--debug-frame", type=str, default=None,
                        help="首帧保存到该 PNG 路径 (诊断渲染输出)")
    args = parser.parse_args()

    tracker = DirectMuscleTrackerMujoco(use_scale_map=True)
    model, data = tracker.model, tracker.data
    viewer = GlfwViewer(model, args.width, args.height,
                        title="Muscle-driven walking (284 muscles)")

    # 骨骼视觉淡化 (与离屏视频一致)
    try:
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "MatBone")
        if mid >= 0:
            model.mat_rgba[mid] = [0.85, 0.85, 0.88, 0.55]
    except Exception:
        pass

    motion = SkeletonMotion.from_file(args.motion)
    dof_pos_ref = local_rotation_to_dof(motion.local_rotation).float()
    root_pos_ref = motion.root_translation
    root_rot_ref = motion.global_rotation[:, 0]
    T = min(dof_pos_ref.shape[0], args.max_frames)
    root_vel = torch.zeros_like(root_pos_ref)
    if T > 1:
        root_vel[1:] = (root_pos_ref[1:] - root_pos_ref[:-1]) * float(motion.fps)
    substeps = max(1, round(int(1.0 / model.opt.timestep) / float(motion.fps)))

    kp = tracker.kp
    kd = tracker.kd

    tracker._set_root(root_pos_ref[0].numpy(), root_rot_ref[0].numpy(), root_vel[0].numpy())
    data.qpos[tracker.qpos_adr] = dof_pos_ref[0].numpy()
    data.qvel[tracker.dof_adr] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, args.height, args.width)
    opt = mujoco.MjvOption()

    last_act = None
    a = torch.zeros(tracker.n_muscles)
    step = 0
    print("窗口已打开 (左键旋转 / 右键平移 / 滚轮缩放 / 关窗退出)。")
    while not viewer.should_close():
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
        a, tau_muscle = optimize_act(JtA, b, tau_des[None], method="lbfgs",
                                     max_iter=args.max_iter, last_act=last_act)
        a = a[0]; tau_muscle = tau_muscle[0]
        last_act = a.detach()
        data.qfrc_applied[tracker.dof_adr] = tau_muscle.numpy()
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)

        if step % args.render_every == 0:
            mujoco.mj_forward(model, data)
            viewer.update_camera(data.xpos[1])
            renderer.update_scene(data, camera=viewer.cam, scene_option=opt)
            # 284 肌肉线 (灰->红, 半径随激活) — 与离屏视频同款
            p_world, n_wp = tracker.compute_muscle_points()
            anp = a.numpy()
            for m in range(p_world.shape[0]):
                t = anp[m]
                col = tracker._activation_color(t)
                rad = max(0.0015, 0.008 * (0.30 + 0.70 * t))
                W = int(n_wp[m])
                for w in range(0, W - 1, max(1, args.muscle_stride)):
                    g = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                                        np.zeros(3), np.zeros(3), np.eye(3).flatten(), col)
                    mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), rad,
                                         p_world[m, w], p_world[m, min(w + 1, W - 1)])
                    renderer.scene.ngeom += 1
            img = renderer.render()
            if args.debug_frame and step == 0:
                import imageio
                imageio.imwrite(args.debug_frame, img)
                print(f"[诊断] 首帧均值={img.mean():.1f} (非全黑=渲染正常), 已存 {args.debug_frame}")
            viewer.render(img)
        step += 1

    viewer.close()
    print("已退出 viewer。")


if __name__ == "__main__":
    main()
