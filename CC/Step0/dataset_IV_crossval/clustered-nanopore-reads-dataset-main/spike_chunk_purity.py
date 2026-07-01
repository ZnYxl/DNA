#!/usr/bin/env python3
"""
spike_chunk_purity.py
=====================
诊断 GradHC 在 dataset IV 上塌缩的真正病因 (spike-first, 不改源码)

方法:
    子类 override run()，在 chunk_partitioning() 执行后、clustering_in_chunks() 之前
    停下，dump 每个 chunk 的 GT 簇组成 (用 read→GT 映射回查)。

判定:
    - chunk 纯度高 (每chunk≈单一GT) → 病在 Step2/3 没拆出来 → 调 q/sd
    - chunk 纯度低 (每chunk混多个GT)  → 病在 chunk_partitioning 过度合并 → 调 min_work/allowed_bad

read→GT 映射: 用 self.all_reads[idx] 的序列回查 GT 字典 (GT对齐铁律, 序列做key)

用法:
    python spike_chunk_purity.py \
        --input      /abs/.../01_gradhc_input_blocked.txt \
        --gt         /abs/.../01_gt_seq_to_tag.txt \
        --gradhc_dir /abs/.../GradHC
"""

import os
import sys
import argparse
from collections import Counter, defaultdict


def banner(t):
    print(f"\n{'─'*60}\n  {t}\n{'─'*60}")


def load_gt(gt_path):
    seq_to_tag = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2:
                seq_to_tag[parts[1].upper()] = int(parts[0])
    return seq_to_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',      required=True)
    ap.add_argument('--gt',         required=True)
    ap.add_argument('--gradhc_dir', required=True)
    args = ap.parse_args()

    banner("加载 GT")
    seq_to_tag = load_gt(args.gt)
    print(f"  GT 唯一序列: {len(seq_to_tag):,}")

    # chdir + import (与 pipeline 一致)
    prev = os.getcwd()
    os.chdir(args.gradhc_dir)
    if args.gradhc_dir not in sys.path:
        sys.path.insert(0, args.gradhc_dir)
    from GradHC_clustering import GradHCBasedCluster

    # 子类: 在 chunk_partitioning 后停下做诊断
    diag = {}
    class GradHCDiag(GradHCBasedCluster):
        def run(self):
            self.pre_step()
            self.chunk_partitioning()
            # ── 此处 self.chunks 已生成，self.all_reads 已填充 ──
            diag['chunks'] = self.chunks
            diag['all_reads'] = self.all_reads
            # 不继续跑 Step2/3，诊断完即停
            return "DIAG_STOP"

    banner("跑到 chunk_partitioning 后停下")
    try:
        c = GradHCDiag(args.input, serial=True, export=False)
        c.run()
    finally:
        os.chdir(prev)

    chunks = diag['chunks']
    all_reads = diag['all_reads']

    # ── 分析每个非空 chunk 的 GT 组成 ──
    banner("Chunk 纯度分析")
    chunk_sizes = []
    chunk_n_gt = []          # 每个chunk混了多少个不同GT
    chunk_purity = []        # 每个chunk内最大GT占比
    n_unmatched = 0

    for chunk in chunks:
        if len(chunk) == 0:
            continue
        tags = []
        for idx in chunk:
            seq = all_reads[idx]
            t = seq_to_tag.get(seq.upper())
            if t is not None:
                tags.append(t)
            else:
                n_unmatched += 1
        if not tags:
            continue
        cnt = Counter(tags)
        size = len(tags)
        n_distinct_gt = len(cnt)
        maj_n = cnt.most_common(1)[0][1]
        chunk_sizes.append(size)
        chunk_n_gt.append(n_distinct_gt)
        chunk_purity.append(maj_n / size)

    n_chunks = len(chunk_sizes)
    import statistics as st
    print(f"  非空 chunk 数:        {n_chunks:,}")
    print(f"  未匹配GT的read:       {n_unmatched:,}")
    print(f"  GT 真值簇数:          9,984")
    print()
    sz = sorted(chunk_sizes, reverse=True)
    print(f"  chunk 大小: max={sz[0]}, med={sz[len(sz)//2]}, "
          f"mean={st.mean(chunk_sizes):.1f}, min={sz[-1]}")
    print()
    print(f"  ◆ 每 chunk 混入的不同 GT 簇数 (理想=1):")
    ng = sorted(chunk_n_gt, reverse=True)
    print(f"     max={ng[0]}, p99={ng[int(len(ng)*0.01)]}, "
          f"med={ng[len(ng)//2]}, mean={st.mean(chunk_n_gt):.1f}")
    print()
    print(f"  ◆ 每 chunk 纯度 (最大GT占比, 理想=1.0):")
    pu = sorted(chunk_purity)
    print(f"     min={pu[0]:.3f}, p10={pu[int(len(pu)*0.1)]:.3f}, "
          f"med={st.median(chunk_purity):.3f}, mean={st.mean(chunk_purity):.3f}")
    print()

    # 加权纯度 (按chunk大小)
    total_reads = sum(chunk_sizes)
    weighted_pure = sum(p*s for p, s in zip(chunk_purity, chunk_sizes))
    print(f"  ◆ 加权平均纯度 (按read数): {weighted_pure/total_reads:.4f}")
    print()

    # 巨chunk暴露
    print(f"  ◆ 最大的5个 chunk 的GT组成:")
    big = sorted(zip(chunk_sizes, chunk_n_gt, chunk_purity),
                 key=lambda x: -x[0])[:5]
    for s, n, p in big:
        print(f"     size={s:5d}, 混入GT簇数={n:4d}, 纯度={p:.3f}")

    banner("病因判定")
    mean_gt = st.mean(chunk_n_gt)
    wp = weighted_pure/total_reads
    if wp >= 0.85 and mean_gt <= 3:
        print(f"  → chunk 纯度高 (加权{wp:.2f}, 平均混入{mean_gt:.1f}个GT)")
        print(f"     病因在 Step2/3: chunk内同源但没被精细聚类拆出")
        print(f"     方向: 检查 resonable_chunk 阈值 / 调 q,sd")
    elif wp < 0.6 or mean_gt > 10:
        print(f"  → chunk 纯度低 (加权{wp:.2f}, 平均混入{mean_gt:.1f}个GT)")
        print(f"     病因在 Step1 chunk_partitioning: 过度合并把异源粘一起")
        print(f"     而 Step2/3 只合不拆，救不回来")
        print(f"     方向: 调 chunk_partitioning 的 min_work↑ / allowed_bad↓ (早停)")
    else:
        print(f"  → 中间地带 (加权{wp:.2f}, 平均混入{mean_gt:.1f}个GT)")
        print(f"     两端都有问题，需进一步细分")

    print(f"\n✅ 诊断完成")


if __name__ == '__main__':
    main()