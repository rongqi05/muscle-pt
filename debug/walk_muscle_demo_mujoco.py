"""MuJoCo 版直接肌肉控制 CLI (替代 Isaac Lab, 无需 GPU, CPU 即可)。

技术路线同 Isaac Lab 版: BVH → q_ref → PD 期望力矩 → optimize 284 activation
→ Hill muscle → 50 torque → MuJoCo 刚体仿真。

用法 (仓库根目录):
    # 单段评估 (默认 lbfgs+50, 与 Isaac Lab 生产一致)
    python debug/walk_muscle_demo_mujoco.py --motion data/cmu_bio_npy/009/09_12.npy

    # 批量评估
    python debug/walk_muscle_demo_mujoco.py --motion "data/cmu_bio_npy/*/*.npy" --batch

    # 保存轨迹 (含 q/q_ref/a/tau_des/tau_muscle, 供 plot_activation.py / render_muscle_mujoco.py)
    python debug/walk_muscle_demo_mujoco.py --motion data/cmu_bio_npy/009/09_12.npy --save-npz output/mj_traj.npz
"""

import argparse
import glob
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from protomotions.utils.direct_muscle_mujoco import (  # noqa: E402
    DirectMuscleTrackerMujoco,
    DOF_NAMES,
)

# 患侧/健侧下肢关节 (用于偏瘫不对称指标)
LEG_DOF_IDX = {
    "L": [i for i, n in enumerate(DOF_NAMES) if n.startswith("L_") and
          any(k in n for k in ("Hip", "Knee", "Ankle", "Toe"))],
    "R": [i for i, n in enumerate(DOF_NAMES) if n.startswith("R_") and
          any(k in n for k in ("Hip", "Knee", "Ankle", "Toe"))],
}


def side_error(r, side):
    """单侧下肢跟踪误差 (rad): mean|q - q_ref| 只统计 side 侧的髋/膝/踝 9 个 DOF。"""
    idx = LEG_DOF_IDX[side]
    return float(np.abs(r["q"][:, idx] - r["q_ref"][:, idx]).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=str, required=True, help=".npy SkeletonMotion 文件(或通配)")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--method", type=str, default="lbfgs", choices=["lbfgs", "ls"])
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--muscle-scale", type=float, default=1.0)
    parser.add_argument("--no-scale-map", action="store_true", help="关闭偏弱肌肉 f0 补偿")
    parser.add_argument("--pd-only", action="store_true", help="对照: 直接施加 PD 力矩")
    parser.add_argument("--batch", action="store_true", help="批量评估 --motion 通配的所有段")
    parser.add_argument("--save-npz", type=str, default=None,
                        help="保存轨迹 npz (q/q_ref/a/tau_des/tau_muscle/body_pos/ref/muscle_names)")
    parser.add_argument("--fail-thresh", type=float, default=0.15, help="批量判定失败的角度阈值 (rad)")
    parser.add_argument("--render-video", type=str, default=None,
                        help="渲染肌肉视频到该 mp4 路径 (骨骼 + 按激活上色的肌肉线段)")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--render-fps", type=int, default=60,
                        help="视频帧率 (物理子步插值, 60fps 更顺滑)")
    parser.add_argument("--muscle-stride", type=int, default=1, help="肌肉 waypoint 降采样 (越大越稀, 渲染越快)")
    parser.add_argument("--no-muscles", action="store_true", help="只渲染骨骼, 不画肌肉")
    parser.add_argument("--affected-side", type=str, default=None, choices=["L", "R"],
                        help="偏瘫患侧 (L=左侧, R=右侧); 与 --affected-strength 一起用")
    parser.add_argument("--affected-strength", type=float, default=0.8,
                        help="患侧肌肉 f0 强度倍率 (如 0.8/0.6/0.4)")
    parser.add_argument("--strength-sweep", action="store_true",
                        help="患侧强度扫描 1.0/0.8/0.6/0.4, 输出对称性对比表")
    args = parser.parse_args()

    files = sorted(glob.glob(args.motion))
    assert files, f"未找到 motion 文件: {args.motion}"

    def make_tracker(strength=None):
        """建 tracker; strength 非 None 时对患侧肌肉 f0 乘 strength。"""
        t = DirectMuscleTrackerMujoco(use_scale_map=not args.no_scale_map)
        if strength is not None:
            n = t.set_hemiplegia_strength(args.affected_side, strength)
            print(f"患侧 {args.affected_side}: {n} 条肌肉 f0 x{strength} ({int(strength*100)}%)")
        return t

    if args.strength_sweep:
        # 患侧强度扫描: 同一段 motion, 不同患侧 f0 强度
        sweep = [1.0, 0.8, 0.6, 0.4]
        print("=" * 88)
        print(f"患侧强度扫描: {files[0]}  side={args.affected_side}  "
              f"method={args.method} max_iter={args.max_iter}")
        print(f"{'strength':>9} {'track(°)':>9} {'患侧(°)':>8} {'健侧(°)':>8} "
              f"{'asym(°)':>8} {'tq_match':>9} {'act_mean':>9} {'NaN':>5}")
        print("-" * 88)
        rows = []
        for s in sweep:
            tracker = make_tracker(strength=s)
            r = tracker.track(files[0], max_frames=args.max_frames, method=args.method,
                              max_iter=args.max_iter, kp_scale=args.kp_scale,
                              muscle_scale=args.muscle_scale, pd_only=args.pd_only)
            if args.affected_side is None:
                aff_side, heal_side = "L", "R"
            else:
                aff_side, heal_side = args.affected_side, ("R" if args.affected_side == "L" else "L")
            err_aff = np.degrees(side_error(r, aff_side))
            err_hea = np.degrees(side_error(r, heal_side))
            asym = err_aff - err_hea
            rows.append((s, r, err_aff, err_hea, asym))
            print(f"{s:>9.1f} {np.degrees(r['tracking_err']):>9.2f} {err_aff:>8.2f} "
                  f"{err_hea:>8.2f} {asym:>8.2f} {r['torque_match']:>9.4f} "
                  f"{r['act_mean']:>9.4f} {str(r['nan_flag']):>5}")
            sys.stdout.flush()
        print("-" * 88)
        base = rows[0]
        print("asym = 患侧跟踪误差 - 健侧跟踪误差 (患侧/健侧比例 80% = strength 0.8)")
        print(f"健康基线: {np.degrees(base[1]['tracking_err']):.2f}°, "
              f"患侧 {base[2]:.2f}° / 健侧 {base[3]:.2f}°")
        print("注: 逐帧激活优化会自动提高患侧激活补偿 f0 损失 (act 未饱和前跟踪几乎不变)")
        return

    tracker = make_tracker(
        strength=None if args.affected_side is None else args.affected_strength)

    def run_one(f):
        motion = None
        return tracker.track(f, max_frames=args.max_frames, method=args.method,
                             max_iter=args.max_iter, kp_scale=args.kp_scale,
                             muscle_scale=args.muscle_scale, pd_only=args.pd_only)

    if args.render_video:
        tracker.render_video(files[0], output=args.render_video, max_frames=args.max_frames,
                             method=args.method, max_iter=args.max_iter,
                             kp_scale=args.kp_scale, muscle_scale=args.muscle_scale,
                             fps=args.render_fps, width=args.render_width,
                             height=args.render_height,
                             show_muscles=not args.no_muscles,
                             muscle_stride=args.muscle_stride)
        return

    if args.batch:
        print("=" * 78)
        print(f"批量评估 (MuJoCo): {len(files)} 段  method={args.method} max_iter={args.max_iter} "
              f"scale_map={not args.no_scale_map}")
        print(f"{'#':>3} {'motion':<34} {'err(°)':>8} {'NaN':>4} {'判定':>6}")
        print("-" * 78)
        errs, fails = [], 0
        for i, f in enumerate(files):
            r = run_one(f)
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
        print(f"成功: {ok}/{len(files)}  失败率: {fails/len(files):.1%}  平均跟踪误差: {mean_deg:.2f}°")
    else:
        r = run_one(files[0])
        print("=" * 72)
        print(f"MuJoCo 直接肌肉控制 (method={args.method}, max_iter={args.max_iter})")
        print("=" * 72)
        print(f"mean |q - q_ref|        = {r['tracking_err']:.4f} rad = {np.degrees(r['tracking_err']):.2f}°")
        print(f"mean ||tau_m - tau_des||/||tau_des|| = {r['torque_match']:.4f}")
        print(f"activation mean        = {r['act_mean']:.4f}")
        print(f"activation 饱和占比     = {r['act_sat']:.4f}")
        print(f"nan                    = {r['nan_flag']}")
        if args.affected_side is not None:
            heal_side = "R" if args.affected_side == "L" else "L"
            err_aff = np.degrees(side_error(r, args.affected_side))
            err_hea = np.degrees(side_error(r, heal_side))
            print(f"患侧({args.affected_side})下肢误差 = {err_aff:.2f}°   "
                  f"健侧({heal_side})下肢误差 = {err_hea:.2f}°   "
                  f"不对称 = {err_aff - err_hea:+.2f}°")
        if args.save_npz:
            ref = tracker.kinematic_reference(files[0], max_frames=args.max_frames)
            np.savez(args.save_npz, actual=r["body_pos"], ref=ref, a=r["a"],
                     q=r["q"], q_ref=r["q_ref"],
                     tau_des=r["tau_des"], tau_muscle=r["tau_muscle"],
                     muscle_names=np.array([m.name for m in tracker.ctl.muscle_char.muscles]))
            print(f"轨迹已保存 -> {args.save_npz} (q/q_ref/a/tau_des/tau_muscle/body_pos/ref)")


if __name__ == "__main__":
    main()
