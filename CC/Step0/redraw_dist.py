#!/usr/bin/env python3
"""
redraw_dist.py
==============
独立的「分布保持验证」重绘工具 —— 不重跑 pipeline，直接从已有数据出图。

用途:
  pipeline 的 thinning_verify 改了画图逻辑后，想给已跑过的 p 重新出图/重算指标，
  无需重跑整个 Clover 流程。本脚本复用同一个 thinning_verify 模块，保证口径一致。

两种输入模式（二选一）:

  模式 A —— 从原始 output.txt 复现抽稀（推荐，最严谨）:
    给定原始 output.txt + keep_ratio + seed，脚本完整复现「过滤→全局随机抽稀」，
    得到 all_reads / cleaned，再调验证模块。结果与 pipeline 跑出来的完全一致。

    python redraw_dist.py modeA \\
        --output_txt  /path/to/output.txt \\
        --keep_ratio  0.1 \\
        --ref_len     196 \\
        --len_min     191 --len_max 201 \\
        --dataset     Seq_1D \\
        --png         /path/to/dist_preserve_p0.1.png \\
        --seed        42

  模式 B —— 从已落盘的 tags 文件出图（快速，仅看抽稀后）:
    只用抽稀后的 tags 文件（tag<TAB>seq 或 tag<space>seq）画「抽稀后」分布。
    注意：此模式没有「抽稀前」全量数据，只能画抽稀后单图、不能算 Wasserstein/KS
    （那些需要前后对比）。适合只想快速看一眼抽稀后形状的场景。
    若要完整五项指标，请用模式 A。

    python redraw_dist.py modeB \\
        --tags_file   /path/to/seq1d_tags_reads.txt \\
        --keep_ratio  0.1 \\
        --dataset     Seq_1D \\
        --png         /path/to/after_only_p0.1.png
"""

import argparse
import os
import sys
import random
from collections import Counter

import numpy as np

# 复用公共验证模块（与 pipeline 同口径）
sys.path.insert(0, '/mnt/st_data/liangxinyi/code/CC/Step0')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thinning_verify import verify_distribution_preserved, _ascii_hist


def _load_and_filter(output_txt, len_min, len_max):
    """读 output.txt → 去N + 长度过滤 → [(tag, seq), ...]（与 pipeline step0 一致）。"""
    all_reads = []
    total = n_dropped = len_dropped = 0
    with open(output_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            tag, seq = parts
            if 'N' in seq.upper():
                n_dropped += 1
                continue
            if not (len_min <= len(seq) <= len_max):
                len_dropped += 1
                continue
            all_reads.append((tag, seq))
    print(f"  总行数: {total:,} | 含N剔除: {n_dropped:,} | 长度剔除: {len_dropped:,} "
          f"| 过滤后: {len(all_reads):,}")
    return all_reads


def mode_a(args):
    """从原始 output.txt 复现抽稀，完整五项验证。"""
    print(f"\n=== 模式 A：从 output.txt 复现抽稀（完整验证）===")
    print(f"  output.txt: {args.output_txt}")
    print(f"  keep_ratio: {args.keep_ratio}  seed: {args.seed}")
    print(f"  长度范围:   [{args.len_min}, {args.len_max}]\n")

    all_reads = _load_and_filter(args.output_txt, args.len_min, args.len_max)
    if not all_reads:
        print("  ❌ 过滤后无 reads"); sys.exit(1)

    # 与 pipeline 完全一致的全局随机抽稀
    rng = random.Random(args.seed)
    n_keep = int(round(len(all_reads) * args.keep_ratio))
    cleaned = rng.sample(all_reads, n_keep)
    print(f"  抽稀后: {len(cleaned):,} reads\n")

    verify_distribution_preserved(
        all_reads, cleaned, args.keep_ratio,
        dataset_name=args.dataset, png_path=args.png,
    )


def mode_b(args):
    """仅从抽稀后 tags 文件画「抽稀后」单图（无前后对比）。"""
    print(f"\n=== 模式 B：从 tags 文件出图（仅抽稀后单图）===")
    print(f"  tags_file: {args.tags_file}\n")

    after_counts = Counter()
    with open(args.tags_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 兼容 TAB 或空格分隔
            if '\t' in line:
                tag = line.split('\t', 1)[0]
            else:
                tag = line.split(' ', 1)[0]
            after_counts[tag] += 1

    after_sizes = np.array(list(after_counts.values()), dtype=float)
    print(f"  tags: {len(after_sizes):,} | "
          f"mean={after_sizes.mean():.1f} median={np.median(after_sizes):.0f} "
          f"max={after_sizes.max():.0f} min={after_sizes.min():.0f}\n")

    print(_ascii_hist(after_sizes, label=f"{args.dataset} 抽稀后 (p={args.keep_ratio})"))

    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            p99 = np.percentile(after_sizes, 99)
            capped = after_sizes[after_sizes <= p99]
            span = capped.max() - capped.min()
            if span <= 100:
                lo = int(np.floor(capped.min())); hi = int(np.ceil(capped.max())) + 1
                step = max(1, int(np.ceil((hi - lo) / 50)))
                bins = np.arange(lo, hi + step, step)
            else:
                bins = 50
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(capped, bins=bins, color="#55A868", edgecolor="white", linewidth=0.3)
            ax.axvline(after_sizes.mean(), color="red", ls="--", lw=1.5,
                       label=f"mean={after_sizes.mean():.1f}")
            ax.axvline(np.median(after_sizes), color="orange", ls="--", lw=1.5,
                       label=f"median={np.median(after_sizes):.0f}")
            ax.set_title(f"{args.dataset}  After thinning p={args.keep_ratio} (mean={after_sizes.mean():.1f})")
            ax.set_xlabel("Reads per GT molecule (redundancy)")
            ax.set_ylabel("Number of GT molecules")
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.png, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✅ PNG 已保存: {args.png}")
        except Exception as e:
            print(f"  ⚠️ PNG 绘制失败: {e}")

    print(f"\n  ⚠️ 模式 B 无「抽稀前」数据，未计算 Wasserstein/KS/Pearson。")
    print(f"     需完整五项指标请用模式 A（提供 output.txt）。")


def main():
    parser = argparse.ArgumentParser(description="分布保持验证 - 独立重绘工具")
    sub = parser.add_subparsers(dest="mode", required=True)

    pa = sub.add_parser("modeA", help="从 output.txt 复现抽稀（完整五项验证）")
    pa.add_argument("--output_txt", required=True)
    pa.add_argument("--keep_ratio", type=float, required=True)
    pa.add_argument("--ref_len", type=int, default=196)
    pa.add_argument("--len_min", type=int, required=True)
    pa.add_argument("--len_max", type=int, required=True)
    pa.add_argument("--dataset", default="dataset")
    pa.add_argument("--png", default=None)
    pa.add_argument("--seed", type=int, default=42)

    pb = sub.add_parser("modeB", help="从 tags 文件出图（仅抽稀后单图）")
    pb.add_argument("--tags_file", required=True)
    pb.add_argument("--keep_ratio", type=float, required=True)
    pb.add_argument("--dataset", default="dataset")
    pb.add_argument("--png", default=None)

    args = parser.parse_args()
    if args.mode == "modeA":
        mode_a(args)
    else:
        mode_b(args)


if __name__ == "__main__":
    main()