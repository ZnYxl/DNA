#!/usr/bin/env python3
"""
gradhc_giant_anatomy_spike.py  —  只读 spike(不改源码 / 不写 production / 不部署)
================================================================================
目的:解剖 SkipChunk(纯 LSH)聚类产出的 top-N 超级巨簇,定位 33,983 巨簇成因:
  - 滚雪球误并:几十~上百个 GT 均匀混杂(每个占比都很低),簇内编辑距离爆炸
  - 真·异源高相似:少数 GT 占大头 或 簇内编辑距离很小但跨 GT

对照基线(来自实测,正常同源簇):簇内编辑距离 P50=1, P90=3, P99=10。
  => 巨簇若 P50≫1 = 距离爆炸的杂烩(LSH 签名碰撞硬塞);
     若 P50≈1 但跨多 GT = 真·异源高相似(sd 调不动,得动 payload)。

附带:跑完把「簇结构 + all_reads」dump 到 --dump_path,
     后续 sd_high 扫描等 spike 可直接读盘,跳过 819s 的 pre_step。

用法(GRADHC_DIR 下):
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/GradHC
    python /path/to/gradhc_giant_anatomy_spike.py \
        --gradhc_input ../gradhc_out_p0.2/01_gradhc_input.txt \
        --tag_file /mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d_p0.2/seq1d_tags_reads.txt \
        --q 8 --sd_high 0.40 --top_n 5 --min_reads 5 \
        --dump_path /tmp/skipchunk_clusters_p0.2.pkl \
        2>&1 | tee /tmp/giant_anatomy_p0.2.log
"""

import argparse
import os
import sys
import time
import pickle
import random
from collections import Counter, defaultdict

try:
    from Levenshtein import distance as edist
except ImportError:
    edist = None  # 退化:用 edlib 或纯 python;下方有兜底


def _edist(a, b):
    if edist is not None:
        return edist(a, b)
    # 兜底:edlib
    import edlib
    return edlib.align(a, b, task='distance')['editDistance']


def build_read_to_tags_pool(tag_file):
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


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100))
    return sorted_vals[idx]


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
    ap.add_argument('--top_n', type=int, default=5, help='解剖前 N 大簇')
    ap.add_argument('--min_reads', type=int, default=5)
    ap.add_argument('--n_gt', type=int, default=11826)
    ap.add_argument('--edit_sample', type=int, default=300,
                    help='每个巨簇内随机抽样多少 read 算两两编辑距离(控制耗时)')
    ap.add_argument('--dump_path', type=str, default=None,
                    help='把簇结构+all_reads dump 到此路径,供后续 spike 读盘复用')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    try:
        from GradHC_clustering import GradHCBasedCluster
    except ImportError:
        sys.stderr.write("❌ 请在 GRADHC_DIR 下运行\n")
        sys.exit(1)

    SD_HIGH = args.sd_high

    class SkipChunkGradHC(GradHCBasedCluster):
        def chunk_partitioning(self, *a, **kw):
            n = len(self.all_reads)
            self.chunks = [[] for _ in range(n)]
            self.chunks[0] = list(range(n))
            self.chunk_parent = [0 for _ in range(n)]
            print(f"[SkipChunk] {n:,} reads 并入单一巨 chunk")

        def clustering_given_chunk(self, chunk_rep, sd_high=None, sd_low=0.28):
            if sd_high is None:
                sd_high = SD_HIGH
            return super().clustering_given_chunk(chunk_rep, sd_high=sd_high, sd_low=sd_low)

        def final_clustering(self, sd_high=None, sd_low=0.22, low_work_rate=0.005,
                             high_work_rate=0.03, rounds_before_refresh=8, min_rounds=300):
            if sd_high is None:
                sd_high = SD_HIGH
            return super().final_clustering(
                sd_high=sd_high, sd_low=sd_low, low_work_rate=low_work_rate,
                high_work_rate=high_work_rate, rounds_before_refresh=rounds_before_refresh,
                min_rounds=min_rounds)

    print("=" * 70)
    print(f"  巨簇解剖 spike (SkipChunk, q={args.q}, sd_high={SD_HIGH})")
    print("=" * 70)

    read_to_tags, _ = build_read_to_tags_pool(args.tag_file)

    cluster = SkipChunkGradHC(
        args.gradhc_input, q=args.q, k=args.k, m=args.m, L=args.L,
        distance_threshold=args.dist, serial=True, export=False)

    t0 = time.time()
    cluster.pre_step()
    cluster.chunk_partitioning()
    cluster.clustering_in_chunks()
    cluster.final_clustering()
    print(f"  [聚类完成] 耗时 {time.time()-t0:.1f}s\n")

    # ---- 簇结构:list of [read indices] ----
    clusters = [list(reads) for reads in cluster.C_til.values() if reads]
    clusters.sort(key=len, reverse=True)

    # ---- dump 落盘(供后续 sd_high 扫描 spike 读盘,跳过 819s)----
    if args.dump_path:
        with open(args.dump_path, 'wb') as f:
            pickle.dump({
                'all_reads': cluster.all_reads,           # idx -> read 串
                'clusters': clusters,                     # [[read idx,...], ...]
                'params': {'q': args.q, 'sd_high': SD_HIGH, 'k': args.k,
                           'm': args.m, 'L': args.L, 'dist': args.dist},
            }, f)
        print(f"  [dump] 簇结构已落盘: {args.dump_path}  "
              f"({os.path.getsize(args.dump_path)/1e6:.1f} MB)\n")

    # ---- read 串 -> 主 tag(众数,用于成分判读)----
    read_to_tag_majority = {}
    for r, tags in read_to_tags.items():
        read_to_tag_majority[r] = Counter(tags).most_common(1)[0][0]

    def gt_profile(read_indices):
        c = Counter()
        unmapped = 0
        for idx in read_indices:
            tag = read_to_tag_majority.get(cluster.all_reads[idx])
            if tag is None:
                unmapped += 1
            else:
                c[tag] += 1
        return c, unmapped

    def intra_edit_dist(read_indices, sample):
        """簇内随机抽样 read,两两算编辑距离(去重 read 串避免 0 距离虚高)。"""
        seqs = [cluster.all_reads[i] for i in read_indices]
        if len(seqs) > sample:
            seqs = random.sample(seqs, sample)
        dists = []
        # 两两全算太贵,固定取相邻配对 + 随机配对混合
        for a in range(0, len(seqs) - 1, 2):
            dists.append(_edist(seqs[a], seqs[a + 1]))
        # 再补一批随机配对
        for _ in range(min(len(seqs), 200)):
            i, j = random.randrange(len(seqs)), random.randrange(len(seqs))
            if i != j:
                dists.append(_edist(seqs[i], seqs[j]))
        return sorted(dists)

    print("=" * 70)
    print(f"  解剖 top-{args.top_n} 大簇")
    print("=" * 70)
    print(f"  对照基线(正常同源簇): 簇内编辑距离 P50=1, P90=3, P99=10")
    print()

    for rank, reads in enumerate(clusters[:args.top_n], 1):
        size = len(reads)
        c, unmapped = gt_profile(reads)
        n_gt = len(c)
        if c:
            top10 = c.most_common(10)
            dom_frac = top10[0][1] / size
            top_str = "  ".join(f"{t}({n/size*100:.1f}%)" for t, n in top10[:8])
        else:
            dom_frac = 0
            top_str = "(unmapped)"
        dists = intra_edit_dist(reads, args.edit_sample)
        d_p50, d_p90, d_p99 = pct(dists, 50), pct(dists, 90), pct(dists, 99)

        # 判读标记
        if n_gt >= 20 and dom_frac < 0.30 and d_p50 > 15:
            verdict = "⚠ 滚雪球杂烩(多GT均匀 + 距离爆炸,LSH碰撞硬塞)"
        elif n_gt >= 20 and dom_frac < 0.30 and d_p50 <= 5:
            verdict = "⚠ 真·异源高相似(多GT但距离极小,sd调不动,需动payload)"
        elif dom_frac >= 0.85:
            verdict = "✓ 基本干净(主导GT占大头)"
        else:
            verdict = "? 混合,需细看"

        print(f"  #{rank} size={size:,}  含{n_gt}个GT  主导={dom_frac*100:.1f}%  unmapped={unmapped}")
        print(f"     簇内编辑距离: P50={d_p50}  P90={d_p90}  P99={d_p99}  (基线 1/3/10)")
        print(f"     top GT: {top_str}")
        print(f"     判读: {verdict}")
        print()

    print("=" * 70)
    print("  [总体判读逻辑]")
    print("=" * 70)
    print("  • 巨簇 P50≫1(如>15)+ 多GT均匀  → LSH 签名碰撞把异源 read 硬塞一起。")
    print("      修法:收紧 sd(提 sd_high)或收紧 final_clustering 代表机制。下一步跑 sd 扫描。")
    print("  • 巨簇 P50≈1 + 多GT          → 真·异源高相似,sd 无力。")
    print("      修法:只能在 payload 区做精细判别(代价大,论文可改讲 GradHC 局限)。")
    print("  • 只有 1~2 个巨簇脏、其余干净 → 定点拆巨簇即可,不必动全局。")
    print("=" * 70)


if __name__ == '__main__':
    main()