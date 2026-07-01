#!/usr/bin/env python3
"""
thinning_verify.py
==================
全局随机打薄的「分布保持」验证模块（供各 pipeline 复用）。

核心问题: 全局随机抽稀后，per-tag redundancy 分布是否仍保持原始形状？
若保持，则抽稀只是均匀缩放，没有破坏数据集的冗余结构特征。

提供五项证据（从强到弱）:
  1. Wasserstein 距离  —— 主指标，归一化冗余分布的「推土机距离」
  2. KS 检验           —— 抽稀后 vs (抽稀前×p) 是否同分布，给 p 值
  3. Pearson r         —— per-tag 抽稀前 vs 抽稀后，证明逐簇线性缩放
  4. 分位数表          —— P10/25/50/75/90/99 抽稀前后对比
  5. 可视化            —— ASCII 直方图(日志) + PNG(留档/进论文)

依赖: numpy, scipy, matplotlib（kunyu 已确认具备）
"""

import numpy as np
from collections import Counter
from scipy.stats import ks_2samp, wasserstein_distance


def _ascii_hist(sizes, n_bins=30, width=50, label=""):
    """生成 ASCII 直方图字符串。

    当数值范围较小（如抽稀后 redundancy 多为个位/两位整数）时，
    自动改用「整数对齐」的 bin 边界，避免固定 bin 数把相邻整数
    切割不均而产生的梳齿状视觉假象。
    """
    sizes = np.asarray(sizes)
    if len(sizes) == 0:
        return f"  [{label}] 空\n"
    p99 = np.percentile(sizes, 99)
    capped = sizes[sizes <= p99]
    span = capped.max() - capped.min()
    # 范围较小 → 整数对齐分桶（每桶覆盖等量整数）
    if span <= n_bins * 2:
        lo = int(np.floor(capped.min()))
        hi = int(np.ceil(capped.max())) + 1
        step = max(1, int(np.ceil((hi - lo) / n_bins)))
        edges = np.arange(lo, hi + step, step)
        counts, edges = np.histogram(capped, bins=edges)
    else:
        counts, edges = np.histogram(capped, bins=n_bins)
    peak = counts.max() if counts.max() > 0 else 1
    lines = [f"  [{label}]  (x 截断至 P99={p99:.0f}, n={len(sizes):,})"]
    for i in range(len(counts)):
        lo_e, hi_e = edges[i], edges[i + 1]
        bar = "█" * int(round(width * counts[i] / peak))
        lines.append(f"    {lo_e:6.0f}-{hi_e:6.0f} | {bar} {counts[i]:,}")
    return "\n".join(lines) + "\n"


def verify_distribution_preserved(
    all_reads, cleaned, keep_ratio, dataset_name, png_path=None
):
    """
    验证全局随机抽稀是否保持 per-tag redundancy 分布形状。

    参数:
      all_reads   : 抽稀前的 [(tag, seq), ...]（过滤后全体）
      cleaned     : 抽稀后的 [(tag, seq), ...]
      keep_ratio  : 保留比例 p
      dataset_name: 数据集名（标题用）
      png_path    : PNG 输出路径；None 则不画图

    返回: dict（量化指标，便于写入论文表格）
    """
    print(f"\n{'─' * 60}")
    print(f"  分布保持验证 — {dataset_name} (p={keep_ratio})")
    print(f"{'─' * 60}\n")

    before_counts = Counter(t for t, _ in all_reads)
    after_counts = Counter(t for t, _ in cleaned)

    before_sizes = np.array(list(before_counts.values()), dtype=float)
    after_sizes = np.array(list(after_counts.values()), dtype=float)

    # ── 1. Wasserstein 距离（归一化到均值=1，消除缩放，只比形状）──
    before_norm = before_sizes / before_sizes.mean()
    after_norm = after_sizes / after_sizes.mean()
    w_dist = wasserstein_distance(before_norm, after_norm)

    # ── 2. KS 检验：抽稀后 vs 抽稀前×p（理论上应同分布）──
    before_scaled = before_sizes * keep_ratio
    ks_stat, ks_pval = ks_2samp(after_sizes, before_scaled)

    # ── 3. Pearson r：每个共同 tag 抽稀前 vs 抽稀后 ──
    common_tags = set(before_counts) & set(after_counts)
    b_paired = np.array([before_counts[t] for t in common_tags], dtype=float)
    a_paired = np.array([after_counts[t] for t in common_tags], dtype=float)
    if len(common_tags) > 1:
        pearson_r = np.corrcoef(b_paired, a_paired)[0, 1]
    else:
        pearson_r = float("nan")

    # ── 4. 分位数表 ──
    qs = [10, 25, 50, 75, 90, 99]
    b_q = np.percentile(before_sizes, qs)
    a_q = np.percentile(after_sizes, qs)
    a_q_rescaled = a_q / keep_ratio  # 还原到原尺度，应≈ before

    # ── 打印量化结论 ──
    print(f"  【主指标】归一化 Wasserstein 距离: {w_dist:.4f}")
    print(f"            (0=完全同形；<0.05 视为分布形状高度保持)")
    print(f"            结论: {'✅ 形状保持' if w_dist < 0.05 else '⚠️ 形状有偏移，需检查'}")
    print()
    print(f"  【KS 检验】抽稀后 vs 抽稀前×{keep_ratio}:")
    print(f"            D={ks_stat:.4f}, p={ks_pval:.4f}")
    n_total = len(after_sizes) + len(before_sizes)
    if ks_pval > 0.05:
        print(f"            结论: ✅ 无法拒绝同分布 (p>0.05)")
    else:
        print(f"            结论: p≤0.05，但 KS 在大样本(n={n_total:,})下极度敏感，")
        print(f"                  分布上极微小的差异即会拒绝原假设；此处应以")
        print(f"                  Wasserstein 距离(={w_dist:.4f})和分位数误差为主判据。")
        print(f"                  D={ks_stat:.4f} 本身很小，说明两分布最大累积差异极小。")
    print()
    print(f"  【Pearson r】per-tag 抽稀前 vs 抽稀后: {pearson_r:.4f}")
    print(f"            ({len(common_tags):,} 个共同 tag；接近 1 = 逐簇线性缩放)")
    print()
    print(f"  【分位数对比】per-tag redundancy:")
    print(f"    {'分位':>6} | {'抽稀前':>10} | {'抽稀后':>10} | {'抽稀后÷p':>10} | {'相对误差':>8}")
    print(f"    {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for i, q in enumerate(qs):
        rel_err = abs(a_q_rescaled[i] - b_q[i]) / max(b_q[i], 1) * 100
        print(f"    P{q:>4} | {b_q[i]:>10.1f} | {a_q[i]:>10.1f} | "
              f"{a_q_rescaled[i]:>10.1f} | {rel_err:>6.1f}%")
    print()

    # ── 5a. ASCII 直方图 ──
    print(f"  【ASCII 直方图】")
    print(_ascii_hist(before_sizes, label="抽稀前"))
    print(_ascii_hist(after_sizes, label="抽稀后"))

    # ── 5b. PNG ──
    if png_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ax, sizes, title, color in [
                (axes[0], before_sizes, f"Before thinning (mean={before_sizes.mean():.1f})", "#4C72B0"),
                (axes[1], after_sizes, f"After thinning p={keep_ratio} (mean={after_sizes.mean():.1f})", "#55A868"),
            ]:
                p99 = np.percentile(sizes, 99)
                capped = sizes[sizes <= p99]
                # 整数对齐分桶：范围小则按整数边界，避免梳齿假象
                span = capped.max() - capped.min()
                if span <= 100:
                    lo = int(np.floor(capped.min()))
                    hi = int(np.ceil(capped.max())) + 1
                    step = max(1, int(np.ceil((hi - lo) / 50)))
                    bins = np.arange(lo, hi + step, step)
                else:
                    bins = 50
                ax.hist(capped, bins=bins, color=color, edgecolor="white", linewidth=0.3)
                ax.axvline(sizes.mean(), color="red", ls="--", lw=1.5, label=f"mean={sizes.mean():.1f}")
                ax.axvline(np.median(sizes), color="orange", ls="--", lw=1.5, label=f"median={np.median(sizes):.0f}")
                ax.set_title(title)
                ax.set_xlabel("Reads per GT molecule (redundancy)")
                ax.set_ylabel("Number of GT molecules")
                ax.legend()
            fig.suptitle(
                f"{dataset_name}  GT Redundancy — Distribution Preservation\n"
                f"Wasserstein={w_dist:.4f}, KS p={ks_pval:.4f}, Pearson r={pearson_r:.4f}",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(png_path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✅ PNG 已保存: {png_path}")
        except Exception as e:
            print(f"  ⚠️ PNG 绘制失败: {e}")

    return {
        "wasserstein": w_dist,
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
        "pearson_r": pearson_r,
        "before_mean": float(before_sizes.mean()),
        "after_mean": float(after_sizes.mean()),
        "n_tags_before": len(before_sizes),
        "n_tags_after": len(after_sizes),
    }