"""渲染 PD walking demo 为 MP4/GIF(mujoco 离屏渲染)。

重放 output/walk_pd_traj.npz 记录的根姿态 + 关节角,渲染 bio.xml 人体网格。

运行方式(仓库根目录):
    PYTHONPATH=. python debug/render_walk_video.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mujoco  # noqa: E402

BIO_XML = os.path.join(REPO_ROOT, "protomotions", "data", "assets", "mjcf", "bio.xml")
TRAJ = os.path.join(REPO_ROOT, "output", "walk_pd_traj.npz")
OUT_MP4 = os.path.join(REPO_ROOT, "output", "walk_pd.mp4")
OUT_GIF = os.path.join(REPO_ROOT, "output", "walk_pd.gif")

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


def main():
    import imageio

    model = mujoco.MjModel.from_xml_path(BIO_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)

    d = np.load(TRAJ)
    root_pos = d["root_pos"]  # (T,3)
    root_rot = d["root_rot"]  # (T,4) wxyz
    q = d["q"]  # (T,50) sim order
    sim_joints = list(d["sim_joints"])

    # sim order -> bio.xml joint qpos 地址
    jid_map = []
    for name in sim_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        jid_map.append(model.jnt_qposadr[jid])

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.2
    cam.azimuth = 90.0
    cam.elevation = 10.0  # 正值=相机在目标上方俯视(负值会仰拍)

    frames = []
    stride = max(1, len(root_pos) // 120)
    for i in range(0, len(root_pos), stride):
        data.qpos[:] = 0.0
        data.qpos[0:3] = root_pos[i]
        data.qpos[3:7] = root_rot[i]  # wxyz
        for k, name in enumerate(sim_joints):
            data.qpos[jid_map[k]] = q[i, k]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = root_pos[i] + [0, 0, 0.0]  # 对准骨盆(身体中心),不要 +0.9(会高过头顶)
        renderer.update_scene(data, camera=cam)
        frames.append(np.ascontiguousarray(renderer.render()))

    print(f"frames: {len(frames)}")
    imageio.mimsave(OUT_MP4, frames, fps=30)
    print(f"saved {OUT_MP4}")
    gif_frames = [frames[i] for i in range(0, len(frames), 2)]
    imageio.mimsave(OUT_GIF, gif_frames, fps=15)
    print(f"saved {OUT_GIF}")


if __name__ == "__main__":
    main()
