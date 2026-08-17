#!/usr/bin/env python3
"""下载 CMU MoCap 动作捕捉数据集 (BVH 格式) 到本地。

数据来源: GitHub 仓库 una-dinosauria/cmu-mocap
  (CMU Motion Capture Database 的 BVH 转换版, Bruce Hahne / cgspeed 2010 release)

每个 subject 一个目录, 内含 <subject>_<编号>.bvh (120fps, 首帧为合成 T-pose)。

用法:
    python data/scripts/download_cmu_mocap.py                        # 下载默认 locomotion subject
    python data/scripts/download_cmu_mocap.py --subjects 08 09 35 91 # 指定多个 subject
    python data/scripts/download_cmu_mocap.py --all                  # 下载全部 113 个 subject (非常大, 慎用)
    python data/scripts/download_cmu_mocap.py --out /path/to/dir     # 自定义输出目录

默认输出目录: <仓库根>/data/cmu_mocap
支持断点续传: 已存在且非空的文件会跳过。
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

REPO = "una-dinosauria/cmu-mocap"
BRANCH = "master"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data"
API_BASE = f"https://api.github.com/repos/{REPO}/contents/data"

# 默认精选 subject (走/跑/慢跑/转弯等基本 locomotion, 适合肌肉运动训练起步)
# 编号需为三位数 (对应 GitHub 目录名, 如 008, 009, 035, 091, 104, 105)。
# 如需更多, 用 --subjects 指定 (参考 cmu-mocap-index-text.txt 的 subject 描述),
# 或 --all 下载全部。
DEFAULT_SUBJECTS = ["008", "009", "035", "091", "104", "105"]

WORKERS = 8          # 并发下载线程数
RETRIES = 4          # 单个文件重试次数


def get_subject_files(subject: str, manifest_dir: str):
    """通过 GitHub API 获取某 subject 的 bvh 文件列表 (带本地缓存, 避免触发 API 限流)。"""
    subject = subject.zfill(3)  # 统一为三位数, 如 8 -> 008
    cache = os.path.join(manifest_dir, f"{subject}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    url = f"{API_BASE}/{subject}"
    files = []
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.load(resp)
            files = sorted(
                x["name"] for x in data if x["name"].endswith(".bvh")
            )
            if not files:
                raise IOError(f"subject {subject} has no bvh files")
            os.makedirs(manifest_dir, exist_ok=True)
            with open(cache, "w") as f:
                json.dump(files, f)
            break
        except Exception as e:
            print(f"  [warn] 获取 subject {subject} 文件列表失败: {e} (第 {attempt + 1} 次)")
            time.sleep(2 * (attempt + 1))
    return files


def download_file(subject: str, name: str, out_root: str):
    """下载单个 bvh 文件, 支持断点续传与重试。"""
    subject = subject.zfill(3)  # 统一为三位数, 如 8 -> 008
    dest_dir = os.path.join(out_root, subject)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, name)

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip", name

    url = f"{RAW_BASE}/{subject}/{name}"
    for attempt in range(RETRIES):
        try:
            tmp = dest + ".part"
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) == 0:
                raise IOError("下载为空文件")
            os.replace(tmp, dest)
            return "ok", name
        except Exception as e:
            if attempt == RETRIES - 1:
                return "fail", f"{name} ({e})"
            time.sleep(1.5 * (attempt + 1))
    return "fail", name


def main():
    parser = argparse.ArgumentParser(description="下载 CMU MoCap BVH 数据集")
    parser.add_argument(
        "--subjects", nargs="+", type=str, default=None,
        help="要下载的 subject 编号列表, 如: 08 09 35 91",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="下载全部 subject (非常大, 慎用)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help=f"输出目录 (默认: <仓库根>/data/cmu_mocap)",
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS,
        help=f"并发下载线程数 (默认 {WORKERS})",
    )
    args = parser.parse_args()

    if args.all:
        # 获取全部 subject 列表 (一次 API 请求, 目录第一层)
        try:
            with urllib.request.urlopen(API_BASE, timeout=30) as resp:
                data = json.load(resp)
            subjects = sorted(
                x["name"] for x in data if x["type"] == "dir"
            )
        except Exception as e:
            print(f"获取全部 subject 列表失败: {e}")
            sys.exit(1)
    elif args.subjects:
        subjects = sorted(s.zfill(3) for s in args.subjects)
    else:
        subjects = DEFAULT_SUBJECTS

    if args.out:
        out_root = args.out
    else:
        out_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cmu_mocap",
        )

    os.makedirs(out_root, exist_ok=True)
    manifest_dir = os.path.join(out_root, "_manifest")
    os.makedirs(manifest_dir, exist_ok=True)

    print(f"输出目录: {out_root}")
    print(f"计划下载 {len(subjects)} 个 subject: {', '.join(subjects)}")

    tasks = []
    for subject in subjects:
        files = get_subject_files(subject, manifest_dir)
        if not files:
            print(f"  [warn] subject {subject} 无文件, 跳过")
            continue
        print(f"  subject {subject}: {len(files)} 个 bvh 文件")
        for name in files:
            tasks.append((subject, name))

    total = len(tasks)
    print(f"共 {total} 个文件, 开始并发下载 (workers={args.workers})...")

    ok = skip = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_file, s, n, out_root) for s, n in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            status, name = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
            if i % 20 == 0 or i == total:
                el = time.time() - t0
                print(f"  进度 {i}/{total}  ok={ok} skip={skip} fail={fail}  用时 {el:.0f}s")

    print(f"\n完成: 成功 {ok}, 跳过(已存在) {skip}, 失败 {fail}")
    if fail:
        print("有失败文件, 重新运行本脚本即可续传重试。")
    print(f"数据位于: {out_root}")


if __name__ == "__main__":
    main()
