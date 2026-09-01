#!/usr/bin/env python3
"""病人运动 PD-only MuJoCo demo。

输入: data/patient_bio_npy/<patient>/<patient>_<side>_segNNN.npy
      (由 convert_patient_bvh_to_bio.py 生成)
功能: PD-only 跟踪评估 (无肌肉优化, 对照基线) + 可选渲染视频。

用法 (仓库根目录):
    # 列出某病人的片段
    PYTHONPATH=. python debug/patient_pd_demo.py --patient Patient_001 --list

    # PD-only 评估一段
    PYTHONPATH=. python debug/patient_pd_demo.py --patient Patient_001 --seg 0

    # 渲染视频
    PYTHONPATH=. python debug/patient_pd_demo.py --patient Patient_001 --seg 0 \
        --render-video output/patient_001_seg0_pd.mp4
"""
import argparse
import glob
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from protomotions.utils.direct_muscle_mujoco import DirectMuscleTrackerMujoco  # noqa: E402

PATIENT_DIR = os.path.join(REPO_ROOT, "data", "patient_bio_npy")


def list_segments(patient):
    files = sorted(glob.glob(os.path.join(PATIENT_DIR, patient, "*.npy")))
    print(f"{patient} 片段 ({len(files)}):")
    for f in files:
        print(" ", os.path.basename(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="Patient_001")
    parser.add_argument("--seg", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--pd-only", action="store_true", default=True)
    parser.add_argument("--render-video", default=None)
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.list or args.seg is None and not args.render_video:
        list_segments(args.patient)
        return

    files = sorted(glob.glob(os.path.join(PATIENT_DIR, args.patient, "*.npy")))
    assert files, f"未找到 {args.patient} 的转换产物 (先跑 convert_patient_bvh_to_bio.py)"
    motion = files[args.seg]
    side = "L" if "_L_" in os.path.basename(motion) else "R"
    print(f"motion: {motion}  (患侧: {side})")

    tracker = DirectMuscleTrackerMujoco(use_scale_map=True)
    if args.render_video:
        tracker.render_video(motion, output=args.render_video,
                             max_frames=args.max_frames, pd_only=True,
                             kp_scale=args.kp_scale)
        return

    r = tracker.track(motion, max_frames=args.max_frames, pd_only=True,
                      kp_scale=args.kp_scale)
    print(f"PD-only 跟踪误差: {np.degrees(r['tracking_err']):.2f}°  nan={r['nan_flag']}")


if __name__ == "__main__":
    main()
