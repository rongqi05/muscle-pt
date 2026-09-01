#!/usr/bin/env python3
"""拆分病人 BVH 长序列为步态片段。

输入: data/patient_bvh/bvh/Patient_XXX_<Side>Paretic_Sequence.bvh (100fps, 90 秒)
      + Patients_Summary_Sequence.xlsx (患侧/身高/病灶)
输出: data/patient_bvh/splits/<patient>_<side>_segNNN.bvh (片段)
      data/patient_bvh/splits/manifest.csv (清单: patient/side/seg/frames/speed)

切分逻辑: 根平移速度滑窗 (窗长 --window 秒, 步长 --stride 秒),
丢弃平均速度 < --min-speed (m/s) 的静立段。

用法:  PYTHONPATH=. python data/scripts/split_patient_bvh.py [--window 5 --stride 2.5]
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.scripts.lafan_utils import read_bvh  # noqa: E402

BVH_DIR = os.path.join(REPO_ROOT, "data", "patient_bvh", "bvh")
XLSX = os.path.join(BVH_DIR, "Patients_Summary_Sequence.xlsx")
OUT_DIR = os.path.join(REPO_ROOT, "data", "patient_bvh", "splits")


def read_summary_xlsx():
    """{Patient_XXX: {side: Left/Right, height_mm: int, lesion: str}}"""
    import openpyxl
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb.active
    out = {}
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    for r in rows[1:]:
        if r[0] is None:
            continue
        out[str(r[0])] = {
            "age": r[1], "height_mm": float(r[2]), "weight": r[3],
            "lesion": r[4], "side": str(r[5]).strip(),
        }
    return out


def parse_bvh_raw(path):
    """返回 (hierarchy_text, motion_lines, frame_time)。"""
    txt = open(path).read()
    hier = txt.split("MOTION")[0] + "MOTION\n"
    m = txt.split("MOTION")[1].strip().splitlines()
    n_frames = int(re.search(r"Frames:\s*(\d+)", m[0]).group(1))
    ft = float(re.search(r"Frame Time:\s*([\d\.]+)", m[1]).group(1))
    return hier, m[2:2 + n_frames], ft


def write_bvh(path, hierarchy, motion_lines, frame_time):
    header = hierarchy.rstrip("\n") + f"\nFrames: {len(motion_lines)}\nFrame Time: {frame_time}\n"
    with open(path, "w") as f:
        f.write(header)
        for ln in motion_lines:
            f.write(ln + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=float, default=5.0, help="片段时长 (秒)")
    parser.add_argument("--stride", type=float, default=2.5, help="滑窗步长 (秒)")
    parser.add_argument("--min-speed", type=float, default=0.20, help="最小平均速度 (m/s)")
    parser.add_argument("--patients", nargs="+", default=None, help="只处理指定病人, 如 Patient_001")
    args = parser.parse_args()

    summary = read_summary_xlsx()
    files = sorted(glob.glob(os.path.join(BVH_DIR, "Patient_*.bvh")))
    if args.patients:
        files = [f for f in files if any(p in os.path.basename(f) for p in args.patients)]
    print(f"病人 BVH: {len(files)} 个")

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "side", "segment", "frames", "duration_s", "speed_m_s", "height_mm"])
        for bvh_path in files:
            base = os.path.basename(bvh_path)
            pid = base.split("_")[0] + "_" + base.split("_")[1]  # Patient_XXX
            side = "L" if "LeftParetic" in base else "R"
            meta = summary.get(pid, {})
            anim = read_bvh(bvh_path)
            pos = anim.pos[:, 0] / 100.0          # cm -> m, 根位置 (T,3)
            fps = anim.fps
            win = int(round(args.window * fps))
            step = int(round(args.stride * fps))
            hier, lines, ft = parse_bvh_raw(bvh_path)
            n_seg = 0
            for s in range(0, len(pos) - win + 1, step):
                seg_pos = pos[s:s + win]
                speed = float(np.linalg.norm(seg_pos[-1] - seg_pos[0]) / (win / fps))
                if speed < args.min_speed:
                    continue
                out = os.path.join(OUT_DIR, f"{pid}_{side}_seg{n_seg:03d}.bvh")
                write_bvh(out, hier, lines[s:s + win], ft)
                w.writerow([pid, side, n_seg, win, win / fps, round(speed, 3),
                            meta.get("height_mm", "")])
                n_seg += 1
            print(f"{pid} ({side}): {n_seg} 段")
    print(f"\n完成 -> {manifest_path}")


if __name__ == "__main__":
    main()
