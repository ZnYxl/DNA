#!/usr/bin/env python3
"""
plot_redundancy_multi.py —— 多数据集 GT 冗余分布（只读，一口气画多张）
============================================================
每个 GT(BWA tag) 长度过滤后有多少条 reads = 测序冗余/覆盖深度。

数据集参数（已服务器实测确认）:
    数据集     目录                              ref_len  过滤范围
    PE_AYB     PE_AYB                            117      [112,122]
    Seq_1D     Sequencing_data_first_dimension   196      [191,201]
    id20       data-nbt17                        177      [172,182]   ← 原始读长(含引物)

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0
    python plot_redundancy_multi.py                 # 跑全部三个
    python plot_redundancy_multi.py PE_AYB id20     # 只跑指定的
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

ROOT = '/mnt/st_data/liangxinyi/code/CC/Step0'

# 数据集配置: 名字 -> (目录, ref_len)
DATASETS = {
    'PE_AYB':  ('PE_AYB',                          117),
    'Seq_1D':  ('Sequencing_data_first_dimension', 196),
    'id20':    ('data-nbt17',                      177),
}

OUT_DIR = os.path.join(ROOT, 'redundancy_plots')
os.makedirs(OUT_DIR, exist_ok=True)


def count_redundancy(output_txt, ref_len):
    len_min, len_max = ref_len - 5, ref_len + 5
    tag_count = Counter()
    total = n_drop = len_drop = 0
    with open(output_txt) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            total += 1
            p = line.split('\t', 1)
            if len(p) != 2:
                continue
            tag, seq = p
            if 'N' in seq.upper():
                n_drop += 1
                continue
            if not (len_min <= len(seq) <= len_max):
                len_drop += 1
                continue
            tag_count[tag] += 1
    return tag_count, total, n_drop, len_drop


def process(name, subdir, ref_len):
    output_txt = os.path.join(ROOT, subdir, 'output.txt')
    if not os.path.exists(output_txt):
        print(f"  ⚠️ [{name}] 跳过，找不到 {output_txt}")
        return None

    print(f"\n{'='*60}\n  [{name}]  ref_len={ref_len}  过滤[{ref_len-5},{ref_len+5}]\n{'='*60}")
    tag_count, total, n_drop, len_drop = count_redundancy(output_txt, ref_len)
    red = np.array(list(tag_count.values()))

    print(f"  总行数 {total:,}  含N剔除 {n_drop:,}  长度不合格 {len_drop:,}")
    print(f"  唯一 GT(tag) 数: {len(red):,}")
    print(f"  mean={red.mean():.1f}  median={int(np.median(red))}  "
          f"min={red.min()}  max={red.max()}  std={red.std():.1f}")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:<2d} = {int(np.percentile(red, p))}")

    np.save(os.path.join(OUT_DIR, f'redundancy_{name}.npy'), red)
    return red


def main():
    sel = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    results = {}
    for name in sel:
        if name not in DATASETS:
            print(f"  未知数据集: {name}（可选: {list(DATASETS.keys())}）")
            continue
        subdir, ref_len = DATASETS[name]
        red = process(name, subdir, ref_len)
        if red is not None:
            results[name] = red

    if not results:
        print("\n  没有可画的数据集。")
        return

    # ── 画图：每个数据集一个子图 ──
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
    if n == 1:
        axes = [axes]
    colors = {'PE_AYB': '#4C72B0', 'Seq_1D': '#55A868', 'id20': '#C44E52'}

    for ax, (name, red) in zip(axes, results.items()):
        # 95 分位截断，避免长尾压扁主体
        cap = int(np.percentile(red, 99))
        bins = np.arange(0, cap + 5, max(1, cap // 50))
        ax.hist(np.clip(red, 0, cap), bins=bins,
                color=colors.get(name, '#888888'), edgecolor='white', alpha=0.85)
        ax.axvline(red.mean(), color='red', ls='--', lw=1.5,
                   label=f'mean={red.mean():.1f}')
        ax.axvline(np.median(red), color='orange', ls='--', lw=1.5,
                   label=f'median={int(np.median(red))}')
        ax.set_xlabel('Reads per GT molecule (redundancy)')
        ax.set_ylabel('Number of GT molecules')
        ax.set_title(f'{name}  GT Redundancy\n(n={len(red):,} molecules, x capped at P99={cap})')
        ax.legend()
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'redundancy_all.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ 合并图: {out_png}")
    print(f"  ✅ 各数据集 .npy: {OUT_DIR}/redundancy_<name>.npy")


if __name__ == '__main__':
    main()