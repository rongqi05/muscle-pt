#!/usr/bin/env python3
"""回放训练好的 PD 专家策略, 采集刚体轨迹, 用 matplotlib 渲染 GIF (绕过 Isaac Sim 渲染器)。

背景: RTX 5060 上 Isaac Sim 渲染窗口/离屏渲染不可用 (Blackwell 不兼容),
且 8GB 显存跑肌肉 Jacobian 紧张。本脚本:
  1. headless 加载 checkpoint (走 teacher 评估路径)
  2. 跑 max_eval_steps 步, 记录策略实际刚体位置 (get_bodies_state)
  3. 同时取参考动作 global_translation
  4. matplotlib 渲染 GIF (策略=蓝, 参考=红) 与关键帧 PNG

用法 (仓库根目录):
    PYTHONPATH=. python debug/replay_policy.py \
        --checkpoint results/walking_pd_expert/last.ckpt \
        --steps 200 --gif output/replay.gif
"""
import argparse
import os
import sys

# 必须在导入 isaacsim 前设置显存优化, 减少肌肉 Jacobian OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="训练好的 checkpoint 路径")
    parser.add_argument("--steps", type=int, default=200, help="回放步数")
    parser.add_argument("--gif", default="output/replay.gif", help="输出 GIF 路径")
    parser.add_argument("--frames", default="output/replay_frames", help="输出关键帧目录")
    parser.add_argument("--num-envs", type=int, default=4, help="并行环境数 (显存紧张用 1-4)")
    args = parser.parse_args()

    import torch
    from pathlib import Path
    import hydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import numpy as np

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": True})
    simulation_app = app_launcher.app

    from lightning.fabric import Fabric
    from protomotions.agents.ppo.agent import PPO
    # 注册 len/eval 等 OmegaConf resolver (训练/评估同款)
    import protomotions.utils.config_utils  # noqa: F401

    # ---- 加载 checkpoint 的 config ----
    _ckpt = Path(args.checkpoint).resolve()
    _config_path = _ckpt.parent / "config.yaml"
    if not _config_path.exists():
        _config_path = _ckpt.parent.parent / "config.yaml"
    if not _config_path.exists():
        _config_path = _ckpt.parent.parent.parent / "config.yaml"
    print(f"加载训练配置: {_config_path}")
    train_config = OmegaConf.load(_config_path)
    # eval_overrides 合并 (headless False -> True, num_envs -> args)
    if train_config.eval_overrides is not None:
        train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
    train_config.headless = True
    train_config.num_envs = args.num_envs

    fabric: Fabric = instantiate(train_config.fabric)
    fabric.launch()
    env = instantiate(train_config.env, device=fabric.device, simulation_app=simulation_app)
    agent: PPO = instantiate(train_config.agent, env=env, fabric=fabric)
    agent.setup()
    agent.load(_ckpt)

    # ---- 回放循环, 记录轨迹 ----
    agent.eval()
    device = fabric.device
    body_names = env.config.robot.body_names
    n_bodies = len(body_names)
    print(f"机器人刚体数: {n_bodies}, 步数: {args.steps}")

    done_indices = None
    actual_pos = []   # (T, n_bodies, 3)
    ref_pos = []      # (T, n_bodies, 3)
    ref_available = True

    with torch.no_grad():
        for step in range(args.steps):
            obs = agent.handle_reset(done_indices)
            actions = agent.model.act(obs)
            # 走 teacher 激活路径 (与训练时一致)
            try:
                a = env.simulator._update_activation(actions)
                env.simulator._last_activations = a
            except Exception as e:
                print(f"[warn] _update_activation 失败: {e}")
            obs, rewards, dones, terminated, extras = agent.env_step(actions)

            # 策略实际刚体位置
            bodies = env.simulator.get_bodies_state()
            actual_pos.append(bodies.rigid_body_pos[0].cpu().numpy())

            # 参考动作位置
            try:
                motion_ids = env.motion_manager.motion_ids
                motion_times = env.motion_manager.motion_times
                ref_state = env.motion_lib.get_motion_state(
                    motion_ids, motion_times, joint_3d_format="quat"
                )
                ref_pos.append(ref_state.rigid_body_pos[0].cpu().numpy())
            except Exception as e:
                ref_available = False
                print(f"[warn] 获取参考位置失败: {e}")

            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)

            if (step + 1) % 20 == 0:
                print(f"  step {step+1}/{args.steps}")

    actual_pos = np.stack(actual_pos)   # (T, n_bodies, 3)
    if ref_available:
        ref_pos = np.stack(ref_pos)
    else:
        ref_pos = None

    print(f"实际轨迹: {actual_pos.shape}")
    os.makedirs("output", exist_ok=True)
    np.savez(
        "output/replay_traj.npz",
        actual_pos=actual_pos,
        ref_pos=ref_pos if ref_pos is not None else np.zeros((0, n_bodies, 3)),
        body_names=np.array(body_names),
    )
    print("轨迹已保存 output/replay_traj.npz")

    # ---- matplotlib 渲染 GIF ----
    _render_gif(actual_pos, ref_pos, body_names, args.gif, args.frames)

    simulation_app.close()
    print("完成!")


def _render_gif(actual, ref, body_names, gif_path, frames_dir):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from poselib.skeleton.skeleton3d import SkeletonMotion  # noqa
    # 用参考动作的父节点索引 (BIO 骨架), 若不可用则用顺序假设
    parents = _get_bio_parents(body_names)

    T = actual.shape[0]
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    # 等距采样 30 帧
    n_frames = min(T, 30)
    idxs = np.linspace(0, T - 1, n_frames).astype(int)

    def draw_skeleton(ax, gp, parents, color, lw=2.0, alpha=1.0):
        for j in range(1, len(parents)):
            p = int(parents[j])
            if p < 0 or p >= len(gp):
                continue
            ax.plot(
                [gp[p, 0], gp[j, 0]],
                [gp[p, 1], gp[j, 1]],
                [gp[p, 2], gp[j, 2]],
                color=color, linewidth=lw, alpha=alpha,
            )

    frames = []
    all_pos = actual if ref is None else np.concatenate([actual, ref], axis=0)
    c = (all_pos.max(axis=(0, 1)) + all_pos.min(axis=(0, 1))) / 2.0
    r = float((all_pos.max(axis=(0, 1)) - all_pos.min(axis=(0, 1))).max()) * 0.6 + 0.5

    for k, i in enumerate(idxs):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        draw_skeleton(ax, actual[i], parents, "tab:blue", lw=2.2)
        if ref is not None:
            draw_skeleton(ax, ref[i], parents, "tab:red", lw=1.6, alpha=0.7)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_title(f"step {int(i)}  蓝=策略 红=参考")
        ax.view_init(elev=20, azim=-60)
        fpath = os.path.join(frames_dir, f"frame_{k:03d}.png")
        fig.savefig(fpath, dpi=110)
        plt.close(fig)

        # 收集 PNG 帧做 GIF
        from PIL import Image
        im = Image.open(fpath).convert("RGB")
        frames.append(im)
        print(f"  帧 {k}/{n_frames}: {fpath}")

    if frames:
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:],
            duration=100, loop=0,
        )
        print(f"GIF 已保存: {gif_path}")


def _get_bio_parents(body_names):
    """BIO 骨架父子关系 (与 debug/walk_pd_demo.py 一致的 23 刚体)."""
    tree = {
        "Pelvis": -1, "Spine": 0, "Torso": 1, "Neck": 2, "Head": 3,
        "ShoulderL": 2, "ArmL": 5, "ForeArmL": 6, "HandL": 7,
        "ShoulderR": 2, "ArmR": 9, "ForeArmR": 10, "HandR": 11,
        "FemurL": 0, "TibiaL": 13, "TalusL": 14, "FootThumbL": 15, "FootPinkyL": 15,
        "FemurR": 0, "TibiaR": 18, "TalusR": 19, "FootThumbR": 20, "FootPinkyR": 20,
    }
    return [tree.get(n, -1) for n in body_names]


if __name__ == "__main__":
    main()
