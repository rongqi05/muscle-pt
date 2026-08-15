"""渲染 knee demo 为 MP4/GIF(mujoco 离屏渲染,不依赖 Isaac Sim)。

重放 output/knee_demo_traj.npz 里的 L_Knee 角度轨迹,渲染 bio.xml 人体网格。

运行方式(仓库根目录):
    PYTHONPATH=. python debug/render_knee_video.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mujoco  # noqa: E402

BIO_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "mjcf", "bio.xml")
TRAJ = os.path.join(REPO_ROOT, "output", "knee_demo_traj.npz")
OUT_MP4 = os.path.join(REPO_ROOT, "output", "knee_demo.mp4")
OUT_GIF = os.path.join(REPO_ROOT, "output", "knee_demo.gif")


def main():
    import imageio

    model = mujoco.MjModel.from_xml_path(BIO_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)

    # 相机:侧面视角观察左膝
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.75]
    cam.distance = 2.4
    cam.azimuth = 90.0
    cam.elevation = 10.0  # 正值=相机在目标上方俯视(负值会仰拍)

    traj = np.load(TRAJ, allow_pickle=True)
    ext = traj["ext"].item()
    flex = traj["flex"].item()

    knee_qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "L_Knee")]

    frames = []

    def render_phase(log, label):
        q = log["q"]
        # 30 fps 降采样(dt=1/240)
        stride = max(1, len(q) // 45)
        for i in range(0, len(q), stride):
            data.qpos[:] = 0.0
            data.qpos[knee_qadr] = q[i]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            pix = renderer.render()
            frames.append(np.ascontiguousarray(pix))

    render_phase(ext, "extension")
    render_phase(flex, "flexion")

    print(f"total frames: {len(frames)}")
    imageio.mimsave(OUT_MP4, frames, fps=30)
    print(f"saved MP4: {OUT_MP4}")
    # GIF(取更少帧避免过大)
    gif_frames = [frames[i] for i in range(0, len(frames), 2)]
    imageio.mimsave(OUT_GIF, gif_frames, fps=15)
    print(f"saved GIF: {OUT_GIF}")


if __name__ == "__main__":
    main()
