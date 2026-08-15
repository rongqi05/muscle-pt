"""程序化行走步态生成器(不依赖外部数据)。

生成一个合成的人类行走参考轨迹(BIO 骨架 50 维 DOF + 根姿态),
用于验证 PD walking 管线。非真实 mocap 数据,仅用于 pipeline 验证。

输出: output/procedural_gait.npz
  - dof_pos (T, 50)  参考关节角(rad,common 顺序)
  - root_pos (T, 3)  根位置
  - root_rot (T, 4)  根四元数(xyzw)

运行: PYTHONPATH=. python debug/procedural_gait.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
IDX = {n: i for i, n in enumerate(DOF_NAMES)}


def generate_gait(duration=6.0, fps=30.0, speed=1.0, step_cycle=1.1):
    """生成一个周期行走步态。

    Args:
        duration: 总时长(秒)
        fps: 帧率
        speed: 前进速度(m/s)
        step_cycle: 单腿一个步态周期(秒)
    """
    T = int(duration * fps)
    t = np.arange(T) / fps  # 时间

    # 相位:左腿 0 相位起,右腿反相
    phase_L = 2 * np.pi * (t / step_cycle)
    phase_R = phase_L + np.pi

    # 摆动相指示(0=stance, 1=swing): 用 sin 半波
    swing_L = np.clip(np.sin(phase_L), 0, 1) ** 1.5
    swing_R = np.clip(np.sin(phase_R), 0, 1) ** 1.5

    dof = np.zeros((T, 50), dtype=np.float32)

    # ---- 髋关节 x(屈伸):摆动腿屈髋向前,支撑腿伸髋向后 ----
    dof[:, IDX["L_Hip_x"]] = 0.45 * np.sin(phase_L) - 0.15
    dof[:, IDX["R_Hip_x"]] = 0.45 * np.sin(phase_R) - 0.15
    # 髋 y(外展):摆动腿轻微外展
    dof[:, IDX["L_Hip_y"]] = 0.10 * swing_L
    dof[:, IDX["R_Hip_y"]] = 0.10 * swing_R

    # ---- 膝关节:摆动相屈膝,支撑相伸直 ----
    dof[:, IDX["L_Knee"]] = 1.0 * swing_L
    dof[:, IDX["R_Knee"]] = 1.0 * swing_R

    # ---- 踝 x(跖屈/背屈) ----
    dof[:, IDX["L_Ankle_x"]] = 0.25 * swing_L - 0.10
    dof[:, IDX["R_Ankle_x"]] = 0.25 * swing_R - 0.10

    # ---- 手臂:与对侧腿反相摆动 ----
    dof[:, IDX["L_Shoulder_x"]] = 0.35 * np.sin(phase_R)
    dof[:, IDX["R_Shoulder_x"]] = 0.35 * np.sin(phase_L)
    # 手肘轻微弯曲
    dof[:, IDX["L_Elbow_x"]] = 0.3
    dof[:, IDX["R_Elbow_x"]] = 0.3

    # ---- 躯干:轻微前倾 + 反向旋转 ----
    dof[:, IDX["Torso_x"]] = 0.08
    dof[:, IDX["Torso_y"]] = 0.06 * np.sin(phase_L)
    dof[:, IDX["Spine_x"]] = 0.05

    # ---- 根轨迹 ----
    root_pos = np.zeros((T, 3), dtype=np.float32)
    root_pos[:, 0] = speed * t  # 前进方向 x
    root_pos[:, 2] = 0.95 + 0.02 * np.sin(2 * phase_L)  # 垂直起伏
    root_pos[:, 1] = 0.03 * np.sin(phase_L)  # 侧向摆动

    root_rot = np.zeros((T, 4), dtype=np.float32)
    root_rot[:, 3] = 1.0  # 单位四元数 xyzw = [0,0,0,1]

    return dof, root_pos, root_rot


def main():
    dof, root_pos, root_rot = generate_gait()
    out = os.path.join(REPO_ROOT, "output", "procedural_gait.npz")
    np.savez(out, dof_pos=dof, root_pos=root_pos, root_rot=root_rot)
    print(f"generated gait: {dof.shape[0]} frames, {dof.shape[1]} DOFs -> {out}")
    # 简单校验
    knee_range = (dof[:, IDX['L_Knee']].min(), dof[:, IDX['L_Knee']].max())
    hip_range = (dof[:, IDX['L_Hip_x']].min(), dof[:, IDX['L_Hip_x']].max())
    print(f"L_Knee range: {knee_range[0]:.2f} ~ {knee_range[1]:.2f} rad")
    print(f"L_Hip_x range: {hip_range[0]:.2f} ~ {hip_range[1]:.2f} rad")
    print(f"root x range: {root_pos[:,0].min():.2f} ~ {root_pos[:,0].max():.2f} m")


if __name__ == "__main__":
    main()
