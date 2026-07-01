#!/usr/bin/env python3
"""
plot_p10_redundancy.py —— P10 GT 冗余分布（只读 output.txt，画一张直方图）
============================================================
每个 GT(BWA tag) 长度过滤后有多少条 reads = 测序冗余/覆盖深度。

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009
    python plot_p10_redundancy.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

BASE = '/mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009'
OUTPUT_TXT = os.path.join(BASE, 'output.txt')
OUT_DIR = os.path.join(BASE, 'dist_plots')
os.makedirs(OUT_DIR, exist_ok=True)

REF_LEN = 200
LEN_MIN, LEN_MAX = 195, 205

# ── 统计每个 tag 的 reads 数 ──
print("统计 GT 冗余（每个 tag 长度过滤后的 reads 数）...")
tag_count = Counter()
total = n_drop = len_drop = 0
with open(OUTPUT_TXT) as f:
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
        if not (LEN_MIN <= len(seq) <= LEN_MAX):
            len_drop += 1
            continue
        tag_count[tag] += 1

red = np.array(list(tag_count.values()))
print(f"\n  总行数 {total:,}  含N剔除 {n_drop:,}  长度不合格 {len_drop:,}")
print(f"  唯一 GT(tag) 数: {len(red):,}")
print(f"\n  mean={red.mean():.1f}  median={int(np.median(red))}  "
      f"min={red.min()}  max={red.max()}  std={red.std():.1f}")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{p:<2d} = {int(np.percentile(red, p))}")

# ── 画图 ──
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.arange(0, red.max() + 5, 5)
ax.hist(red, bins=bins, color='#4C72B0', edgecolor='white', alpha=0.85)
ax.axvline(red.mean(), color='red', ls='--', lw=1.5, label=f'mean={red.mean():.1f}')
ax.axvline(np.median(red), color='orange', ls='--', lw=1.5,
           label=f'median={int(np.median(red))}')
ax.set_xlabel('Reads per GT molecule (redundancy)')
ax.set_ylabel('Number of GT molecules')
ax.set_title('P10 GT Redundancy Distribution\n(after length filter, before thinning)')
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()

out_png = os.path.join(OUT_DIR, 'p10_redundancy.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
np.save(os.path.join(OUT_DIR, 'gt_redundancy.npy'), red)
print(f"\n  ✅ 图: {out_png}")
print(f"  ✅ 数据: {OUT_DIR}/gt_redundancy.npy")