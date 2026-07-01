#!/usr/bin/env python3
"""
spike_gt_check.py
=================
只读自检：BWA GT(read2ref_q60.tsv) 是否被重复 motif 误导。

逻辑：抽样 size>=5 的簇，算簇内每条 read 到其 GT ref 的归一化 edit distance。
  - 若多数 ED 很小(<5%) → GT 可信，进 Clover
  - 若一批簇 ED 系统性偏大 → BWA 误配，GT 需用 ED 重建

不写任何文件。
用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/stairloop
    python spike_gt_check.py
"""
import os, gzip, random
from collections import defaultdict
import edlib

BASE   = '/mnt/st_data/liangxinyi/code/CC/Step0/stairloop'
FASTQ  = os.path.join(BASE, 'sequencing reads', 'filter_sequences3.fastq')
GT_TSV = os.path.join(BASE, 'read2ref_q60.tsv')
REF    = os.path.join(BASE, 'test_encode.fasta')

N_CLUSTERS = 50      # 抽多少个簇
MIN_SIZE   = 5
SEED       = 42

def load_ref():
    d, rid = {}, None
    with open(REF) as f:
        for line in f:
            if line.startswith('>'):
                rid = line[1:].strip()
            elif rid:
                d[rid] = line.strip(); rid = None
    return d

def load_gt():
    g = {}
    with open(GT_TSV) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2: g[p[0]] = p[1]
    return g

def main():
    ref = load_ref(); gt = load_gt()
    print(f"ref={len(ref):,}  gt={len(gt):,}")

    # ref_id -> 抽中的 read 序列列表
    by_ref = defaultdict(list)
    # 先统计每个 ref 有多少 read，选 size>=MIN_SIZE 的
    ref_count = defaultdict(int)
    for rid, rf in gt.items():
        ref_count[rf] += 1
    big = [r for r, c in ref_count.items() if c >= MIN_SIZE]
    random.seed(SEED)
    chosen = set(random.sample(big, min(N_CLUSTERS, len(big))))
    print(f"size>={MIN_SIZE} 的簇: {len(big):,}，抽 {len(chosen)} 个")

    # 扫 fastq，收集被抽中簇的 reads
    op = gzip.open(FASTQ, 'rt') if FASTQ.endswith('.gz') else open(FASTQ)
    with op as f:
        while True:
            h = f.readline()
            if not h: break
            seq = f.readline().strip(); f.readline(); f.readline()
            rid = h[1:].split()[0]
            rf = gt.get(rid)
            if rf in chosen:
                by_ref[rf].append(seq)

    # 算 ED
    all_ed, bad_clusters = [], []
    for rf, seqs in by_ref.items():
        rseq = ref.get(rf)
        if not rseq: continue
        eds = [edlib.align(s, rseq, task='distance')['editDistance'] / len(rseq)
               for s in seqs]
        med = sorted(eds)[len(eds)//2]
        all_ed.extend(eds)
        if med > 0.10:           # 簇中位 ED >10% → 可疑
            bad_clusters.append((rf, round(med,3), len(seqs)))

    all_ed.sort()
    n = len(all_ed)
    print(f"\n=== 簇内 read→GT-ref 归一化 ED 分布 (n={n}) ===")
    for q, name in [(0.5,'中位'), (0.9,'P90'), (0.95,'P95'), (0.99,'P99')]:
        print(f"  {name}: {all_ed[int(n*q)]:.3f}")
    print(f"  ED<0.05 占比: {100*sum(1 for e in all_ed if e<0.05)/n:.1f}%")
    print(f"  ED>0.10 占比: {100*sum(1 for e in all_ed if e>0.10)/n:.1f}%")

    print(f"\n=== 可疑簇 (中位ED>0.10): {len(bad_clusters)}/{len(by_ref)} ===")
    for rf, med, sz in sorted(bad_clusters, key=lambda x:-x[1])[:10]:
        print(f"  ref={rf}  中位ED={med}  size={sz}")

    print("\n=== 判定 ===")
    if all_ed[int(n*0.95)] < 0.10:
        print("  ✅ GT 干净：P95<10%，BWA 未被 motif 误导，直接进 Clover")
    else:
        print("  ⚠ GT 可疑：高分位 ED 偏大，疑似误配，建议用 ED 重建 GT")

if __name__ == '__main__':
    main()