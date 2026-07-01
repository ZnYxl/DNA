#!/usr/bin/env python3
"""
redraw_thinning_figs.py
=======================
独立重画「分布保持」三张图（p=0.2 / 0.1 / 0.05），不重跑 pipeline。

梳齿根因（已用合成数据验证）
----------------------------
打薄后 after 的 redundancy 是小整数（多在 2~120）。若用「非整数 bin 宽」
（如 0..P99 等分 50 份 → bin 宽≈2.4），有的 bin 框住 3 个整数、相邻 bin 只框
2 个，框 3 个的天然计数更高 → 高低交错成「梳齿」。这是离散数据用非整数 bin 宽
的经典假象，与左右是否对齐无关。

修复：bin 宽强制取整数（且下限=2，防小 p 时被压成 1），每个 bin 框等量整数值
→ 计数均匀、梳齿消失。before 因数值大、相对连续，bin 宽自然取到较大整数，本就平滑。

设计取舍
--------
left/right 各自取最优整数 bin 宽（before≈13, after 3/2/2），不强行同尺度：
  • 「形状一致」由顶部 Wasserstein / KS / Pearson 三指标严格证明，无需靠视觉叠合；
  • 保留真实横轴量级（mean 236 vs 47），直观体现「打薄到 1/p、形状不变」。

数据来源（均为已落盘产物，不依赖重新打薄）
  • before 分布：原始 output.txt 重做同样预处理（去 N + 长度过滤 [191,201]）
  • after  分布：各实验目录 seq1d_tags_reads.txt（tag<TAB>read，每行一条打薄后 read）

用法
----
    python redraw_thinning_figs.py
    python redraw_thinning_figs.py --keep_ratios 0.2 0.1 0.05
    python redraw_thinning_figs.py --output_txt /path/output.txt --exp_root /path/Experiments
"""

import os
import argparse
import numpy as np
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance

# ── 默认路径（与 pipeline_seq1d.py 定义一致）──
DEFAULT_OUTPUT_TXT = "/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/output.txt"
DEFAULT_EXP_ROOT   = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments"

REF_LEN = 196
LEN_MIN = REF_LEN - 5   # 191
LEN_MAX = REF_LEN + 5   # 201

TARGET_NBINS = 45        # 目标 bin 数（实际 bin 宽就近取整，且下限=2）


def load_before_tag_counts(output_txt):
    """重做 pipeline Step 0 预处理（不打薄），返回 per-tag 计数。"""
    counts = Counter()
    n_total = n_n = n_len = 0
    with open(output_txt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            tag, seq = parts
            if "N" in seq.upper():
                n_n += 1
                continue
            if not (LEN_MIN <= len(seq) <= LEN_MAX):
                n_len += 1
                continue
            counts[tag] += 1
    print(f"  [before] output.txt 总行 {n_total:,}, 含N剔 {n_n:,}, 长度剔 {n_len:,}, "
          f"保留 {sum(counts.values()):,} reads / {len(counts):,} tags")
    return counts


def load_after_tag_counts(tags_file):
    """读 seq1d_tags_reads.txt（tag<TAB>read），返回 per-tag 计数。"""
    counts = Counter()
    with open(tags_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tag = line.split("\t", 1)[0]
            counts[tag] += 1
    print(f"  [after ] {tags_file}: {sum(counts.values()):,} reads / {len(counts):,} tags")
    return counts


def int_bin_edges(sizes, target_nbins=TARGET_NBINS, min_width=2):
    """
    整数对齐 bin 边界：bin 宽取整数，每个 bin 框住等量整数值 → 消除梳齿。
    上界取 P99（截长尾）；返回 (edges, step)。

    min_width=2：小 p 时 redundancy 范围很小（如 p=0.05 仅到 ~35），
    若仍按 target_nbins 等分会把 bin 宽压成 1，bin 宽=1 又会出轻微锯齿。
    故强制 bin 宽 ≥ 2（小 span 时自动落到更少 bin），保证平滑。
    """
    p99 = np.percentile(sizes, 99)
    lo = int(np.floor(sizes.min()))
    hi = int(np.ceil(p99)) + 1
    step = max(min_width, int(round((hi - lo) / target_nbins)))
    edges = np.arange(lo, hi + step, step)
    return edges, step


def plot_panel(ax, sizes, title, color):
    """画单个子图：整数 bin 宽；截断 P99 以上长尾（不并入末 bin，避免边界堆积突柱）。"""
    edges, step = int_bin_edges(sizes)
    keep = sizes[(sizes >= edges[0]) & (sizes <= edges[-1])]
    ax.hist(keep, bins=edges, color=color, edgecolor="white", linewidth=0.3)
    ax.axvline(sizes.mean(), color="red", ls="--", lw=1.5, label=f"mean={sizes.mean():.1f}")
    ax.axvline(np.median(sizes), color="orange", ls="--", lw=1.5, label=f"median={np.median(sizes):.0f}")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_title(f"{title}  (bin width={step})")
    ax.set_xlabel("Reads per GT molecule (redundancy)")
    ax.set_ylabel("Number of GT molecules")
    ax.legend()


def redraw_one(keep_ratio, exp_root, before_counts):
    ratio_tag = f"p{keep_ratio}"
    exp_dir = os.path.join(exp_root, f"seq_1d_{ratio_tag}")
    tags_file = os.path.join(exp_dir, "seq1d_tags_reads.txt")
    png_path = os.path.join(exp_dir, f"dist_preserve_{ratio_tag}.png")

    if not os.path.exists(tags_file):
        print(f"  跳过 {ratio_tag}: 不存在 {tags_file}")
        return None

    print(f"\n{'-'*60}\n  重画 {ratio_tag}\n{'-'*60}")
    after_counts = load_after_tag_counts(tags_file)

    before_sizes = np.array(list(before_counts.values()), dtype=float)
    after_sizes = np.array(list(after_counts.values()), dtype=float)

    # ── 指标（与原模块口径一致）──
    w_dist = wasserstein_distance(before_sizes / before_sizes.mean(),
                                  after_sizes / after_sizes.mean())
    _, ks_pval = ks_2samp(after_sizes, before_sizes * keep_ratio)
    common = set(before_counts) & set(after_counts)
    if len(common) > 1:
        b_p = np.array([before_counts[t] for t in common], dtype=float)
        a_p = np.array([after_counts[t] for t in common], dtype=float)
        pearson_r = np.corrcoef(b_p, a_p)[0, 1]
    else:
        pearson_r = float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_panel(axes[0], before_sizes,
               f"Before thinning (mean={before_sizes.mean():.1f})", "#4C72B0")
    plot_panel(axes[1], after_sizes,
               f"After thinning p={keep_ratio} (mean={after_sizes.mean():.1f})", "#55A868")
    fig.suptitle(
        f"Seq_1D  GT Redundancy — Distribution Preservation\n"
        f"Wasserstein={w_dist:.4f}, KS p={ks_pval:.4f}, Pearson r={pearson_r:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  已重画: {png_path}")
    print(f"     Wasserstein={w_dist:.4f}  KS p={ks_pval:.4f}  Pearson r={pearson_r:.4f}")
    return png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep_ratios", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    ap.add_argument("--output_txt", default=DEFAULT_OUTPUT_TXT)
    ap.add_argument("--exp_root", default=DEFAULT_EXP_ROOT)
    args = ap.parse_args()

    print("=" * 60)
    print("  重画分布保持图（不重跑 pipeline，整数 bin 宽消梳齿）")
    print("=" * 60)
    print(f"  output.txt : {args.output_txt}")
    print(f"  exp_root   : {args.exp_root}")
    print(f"  keep_ratios: {args.keep_ratios}")
    print()

    print("  -- 重建 before 分布（一次，三图共用）--")
    before_counts = load_before_tag_counts(args.output_txt)

    done = []
    for kr in args.keep_ratios:
        p = redraw_one(kr, args.exp_root, before_counts)
        if p:
            done.append(p)

    print(f"\n{'='*60}\n  完成，共重画 {len(done)} 张：")
    for p in done:
        print(f"    {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()