#!/usr/bin/env python3
"""
gradhc_skip_chunk_spike.py  —  只读 spike(不改源码 / 不写 production 文件 / 不部署)
================================================================================
核心问题:不靠 chunk_partitioning(已诊断为塌缩元凶),
         只用 GradHC 的 LSH 阶段(clustering_in_chunks + final_clustering, q=8),
         在带引物 196bp 数据上,最终 Purity 能到多少?能否超过 Clover(91%+)?

机制(子类多态,零侵入源码):
  - pre_step() 里 self.chunks = [[idx] for idx in range(N)](每 read 自成 chunk)
  - 覆盖版 chunk_partitioning() 不做公共子串聚链,而是把所有 read union 进 chunk 0:
        chunks[0] = list(range(N)); 其余清空; chunk_parent 全指向 0
    => clustering_in_chunks 看到一个 size=N 的巨 chunk(avg_chunk=N,满足 0.5*avg),
       照常跑 clustering_given_chunk(L=32 轮 LSH+sd),之后 final_clustering 全局聚。
  - 完全复用源码阶段 2/3,只是绕过了塌缩的第一阶段。

评测:与 pipeline step4 同口径(read 串当 key 回查 tag,主导 GT 计纯度),
     并做同样的小簇过滤(min_reads=5)后再算,保证可与 84% 版直接对比。

⚠ 性能提示:对一个 55 万的巨 chunk 跑 32 轮 LSH 比正常分块慢,
   可能十几分钟~半小时。脚本打印各阶段耗时,这是路线 B 的真实代价。

用法(必须在 GRADHC_DIR 下,import 时固定 WORKING_DIR_ALGORITHMS):
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/GradHC
    python /path/to/gradhc_skip_chunk_spike.py \
        --gradhc_input ../gradhc_out_p0.2/01_gradhc_input.txt \
        --tag_file /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d_p0.2/seq1d_tags_reads.txt \
        --q 8 --sd_high 0.40 --min_reads 5 \
        2>&1 | tee /tmp/skip_chunk_p0.2.log

参数对齐 production(q=8, sd_high=0.40),让结果可直接对比 84% 版。
"""

import argparse
import os
import sys
import time
import random
from collections import Counter, defaultdict


def build_read_to_tags_pool(tag_file):
    """read 串 -> [tags](可消耗 pool,与 pipeline step4 同口径)。"""
    read_to_tags = defaultdict(list)
    n = 0
    with open(tag_file) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            tag, read = parts
            read_to_tags[read].append(tag)
            n += 1
    return read_to_tags, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gradhc_input', required=True)
    ap.add_argument('--tag_file', required=True)
    ap.add_argument('--q', type=int, default=8)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--m', type=int, default=40)
    ap.add_argument('--L', type=int, default=32)
    ap.add_argument('--dist', type=int, default=12)
    ap.add_argument('--sd_high', type=float, default=0.40)
    ap.add_argument('--min_reads', type=int, default=5, help='小簇过滤阈值(与 pipeline 对齐)')
    ap.add_argument('--n_gt', type=int, default=11826)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    try:
        from GradHC_clustering import GradHCBasedCluster
    except ImportError:
        sys.stderr.write("❌ 请在 GRADHC_DIR 下运行(cd .../GradHC && python <脚本>)\n")
        sys.exit(1)

    SD_HIGH = args.sd_high

    class SkipChunkGradHC(GradHCBasedCluster):
        """绕过 chunk 公共子串:把全部 read 塞进一个巨 chunk,只跑 LSH 阶段。"""

        def chunk_partitioning(self, *a, **kw):
            # 覆盖:不做公共子串聚链。把所有 read union 进 chunk 0。
            t0 = time.time()
            n = len(self.all_reads)
            # chunks: chunk_rep -> [read indices]。全部并入 0 号。
            self.chunks = [[] for _ in range(n)]
            self.chunks[0] = list(range(n))
            # 并查集:所有 read 的 chunk parent 指向 0
            self.chunk_parent = [0 for _ in range(n)]
            print(f"[SkipChunk] 所有 {n:,} reads 并入单一巨 chunk(绕过公共子串塌缩),"
                  f"耗时 {time.time()-t0:.2f}s")

        # 阶段 2/3 用 production 的 sd_high(覆盖默认,与 84% 版对齐)
        def clustering_given_chunk(self, chunk_rep, sd_high=None, sd_low=0.28):
            if sd_high is None:
                sd_high = SD_HIGH
            return super().clustering_given_chunk(chunk_rep, sd_high=sd_high, sd_low=sd_low)

        def final_clustering(self, sd_high=None, sd_low=0.22, low_work_rate=0.005,
                             high_work_rate=0.03, rounds_before_refresh=8, min_rounds=300):
            if sd_high is None:
                sd_high = SD_HIGH
            return super().final_clustering(
                sd_high=sd_high, sd_low=sd_low,
                low_work_rate=low_work_rate, high_work_rate=high_work_rate,
                rounds_before_refresh=rounds_before_refresh, min_rounds=min_rounds)

        def run_spike(self):
            t = {}
            ts = time.time(); self.pre_step();                 t['pre_step'] = time.time()-ts
            ts = time.time(); self.chunk_partitioning();       t['skip_chunk'] = time.time()-ts
            ts = time.time(); self.clustering_in_chunks();     t['cluster_in_chunk'] = time.time()-ts
            ts = time.time(); self.final_clustering();         t['final'] = time.time()-ts
            return t

    print("=" * 70)
    print("  路线 B spike:绕过 chunk,只用 LSH 阶段(q=%d, sd_high=%.2f)" % (args.q, SD_HIGH))
    print("=" * 70)
    print(f"  gradhc_input : {args.gradhc_input}")
    print(f"  tag_file     : {args.tag_file}")
    print(f"  min_reads    : {args.min_reads}(与 pipeline 对齐)")
    print()

    read_to_tags, n_tag_lines = build_read_to_tags_pool(args.tag_file)
    print(f"  [映射] tag 行数={n_tag_lines:,}  唯一 read={len(read_to_tags):,}")
    print()

    cluster = SkipChunkGradHC(
        args.gradhc_input,
        q=args.q, k=args.k, m=args.m, L=args.L,
        distance_threshold=args.dist,
        serial=True, export=False,
    )

    t_all = time.time()
    timings = cluster.run_spike()
    print()
    print("  [耗时] " + "  ".join(f"{k}={v:.1f}s" for k, v in timings.items())
          + f"  TOTAL={time.time()-t_all:.1f}s")

    # ---- 取最终聚类:C_til[rep] = [read indices],非空即一簇 ----
    clusters = [reads for reads in cluster.C_til.values() if reads]
    sizes_all = sorted((len(c) for c in clusters), reverse=True)

    print()
    print("=" * 70)
    print("  [结果] 最终簇分布(LSH 阶段产出,过滤前)")
    print("=" * 70)
    print(f"  簇数={len(clusters):,}  归簇 reads={sum(sizes_all):,}")
    if sizes_all:
        print(f"  max={sizes_all[0]:,}  median={sizes_all[len(sizes_all)//2]}  min={sizes_all[-1]}")

    # ---- 小簇过滤(与 pipeline step4 对齐)----
    clusters_f = [c for c in clusters if len(c) >= args.min_reads]
    sizes_f = sorted((len(c) for c in clusters_f), reverse=True)
    total_reads_f = sum(sizes_f)
    print()
    print(f"  过滤后(min_reads={args.min_reads}):簇数={len(clusters_f):,}  reads={total_reads_f:,}")
    if sizes_f:
        print(f"  max={sizes_f[0]:,}  median={sizes_f[len(sizes_f)//2]}  min={sizes_f[-1]}")

    # ---- Purity / Coverage(read 串当 key,消耗 pool,与 step4 同口径)----
    pool = {r: list(tags) for r, tags in read_to_tags.items()}
    total_pure = 0
    gt_covered = set()
    for reads in clusters_f:
        tags = []
        for idx in reads:
            read = cluster.all_reads[idx]      # ★ sequence as key
            cand = pool.get(read)
            if cand:
                tags.append(cand.pop())
        if not tags:
            continue
        c = Counter(tags)
        maj_tag, maj_cnt = c.most_common(1)[0]
        total_pure += maj_cnt
        gt_covered.add(maj_tag)

    purity = total_pure / max(total_reads_f, 1)
    coverage = len(gt_covered) / args.n_gt

    print()
    print("=" * 70)
    print("  [核心指标] 路线 B(绕过 chunk,纯 LSH)")
    print("=" * 70)
    print(f"  Purity   : {purity*100:.2f}%   (84% 版 = 84.13%,Clover = 91%+)")
    print(f"  Coverage : {coverage*100:.2f}%   ({len(gt_covered)}/{args.n_gt})")
    print()
    print("=" * 70)
    print("  [判读]")
    print("=" * 70)
    print("  ① Purity > 91% → 路线 B 成立!chunk 阶段是负资产,跳过即修好,论文故事干净。")
    print("  ② 84% < Purity < 91% → 有改善但没到 Clover,需叠加阶段 2/3 调参或换路线 A。")
    print("  ③ Purity ≈ 84% 或更低 → LSH 阶段本身在带引物长 read 上也不行,放弃 B,转路线 A")
    print("     (子类覆盖 chunk_partitioning,把公共子串签名换成跨 payload 的 minimizer)。")
    print("  注:同时看簇数——若簇数远少于 ~11826(GT 数)说明仍在过度合并(欠分割)。")
    print("=" * 70)


if __name__ == '__main__':
    main()