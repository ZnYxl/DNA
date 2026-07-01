#!/usr/bin/env python3
"""
spike_rescue_empty_ref.py
=========================
只读验证：当前无 GT 的 read，反向互补后能否匹配回"空 ref"(被丢的 6055 个)。

逻辑：
  1. 从 read2ref_q60.tsv 得到"已覆盖 ref"集合
  2. 空 ref = 全部 ref - 已覆盖 ref
  3. 抽样无 GT 的 read，翻转后对全 ref 找最近邻(edlib)
  4. 若翻转后能低 ED 匹配到某 ref(尤其空 ref) → 可救回覆盖

注意：这里用暴力最近邻只对抽样 read 做，验证用，不全量。
不写文件。
用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/stairloop
    python spike_rescue_empty_ref.py
"""
import os, gzip, random
import edlib

BASE   = '/mnt/st_data/liangxinyi/code/CC/Step0/stairloop'
FASTQ  = os.path.join(BASE, 'sequencing reads', 'filter_sequences3.fastq')
GT_TSV = os.path.join(BASE, 'read2ref_q60.tsv')
REF    = os.path.join(BASE, 'test_encode.fasta')

SAMPLE_NOGT = 2000     # 抽多少条无 GT read 来验
SEED = 42
_COMP = str.maketrans('ACGTacgt', 'TGCATGCA')
def revcomp(s): return s.translate(_COMP)[::-1]

def load_ref():
    d, rid = {}, None
    with open(REF) as f:
        for line in f:
            if line.startswith('>'): rid = line[1:].strip()
            elif rid: d[rid] = line.strip(); rid = None
    return d

def load_gt_readids():
    ids, covered = set(), set()
    with open(GT_TSV) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                ids.add(p[0]); covered.add(p[1])
    return ids, covered

def nearest(seq, ref_items):
    """暴力最近邻：返回 (best_ref, best_ned)"""
    best_r, best_d = None, 1.0
    for rid, rseq in ref_items:
        d = edlib.align(seq, rseq, task='distance')['editDistance'] / len(rseq)
        if d < best_d:
            best_d, best_r = d, rid
            if d < 0.02: break
    return best_r, best_d

def main():
    ref = load_ref()
    gt_ids, covered = load_gt_readids()
    all_ref = set(ref.keys())
    empty_ref = all_ref - covered
    print(f"全 ref: {len(all_ref):,}")
    print(f"已覆盖 ref: {len(covered):,}")
    print(f"空 ref(被丢): {len(empty_ref):,}")

    ref_items = list(ref.items())  # 暴力最近邻对象（4.5万，每条 read 较慢，故仅抽样）

    # 抽样无 GT 的 read
    rng = random.Random(SEED)
    op = gzip.open(FASTQ, 'rt') if FASTQ.endswith('.gz') else open(FASTQ)
    nogt_reads = []
    with op as f:
        while True:
            h = f.readline()
            if not h: break
            seq = f.readline().strip(); f.readline(); f.readline()
            rid = h[1:].split()[0]
            if rid in gt_ids: continue          # 有 GT 跳过
            if rng.random() < SAMPLE_NOGT/30000: # 无 GT 约占总量，粗略抽
                nogt_reads.append(seq)
            if len(nogt_reads) >= SAMPLE_NOGT: break

    print(f"\n抽样无 GT read: {len(nogt_reads):,}")
    print("(暴力最近邻较慢，请耐心)\n")

    fwd_hit = rc_hit = rc_to_empty = both_miss = 0
    for i, seq in enumerate(nogt_reads):
        if i and i % 200 == 0: print(f"  ...{i}")
        _, fwd_d = nearest(seq, ref_items)
        if fwd_d < 0.05:
            fwd_hit += 1; continue
        rc_r, rc_d = nearest(revcomp(seq), ref_items)
        if rc_d < 0.05:
            rc_hit += 1
            if rc_r in empty_ref: rc_to_empty += 1
        else:
            both_miss += 1

    n = len(nogt_reads)
    print(f"\n=== 无 GT read 救回验证 (n={n}) ===")
    print(f"  正向就能匹配(<0.05): {fwd_hit:,}  ({100*fwd_hit/n:.1f}%)")
    print(f"  翻转后能匹配(<0.05): {rc_hit:,}  ({100*rc_hit/n:.1f}%)")
    print(f"    其中匹配到空ref:   {rc_to_empty:,}")
    print(f"  正反都不匹配(真丢):  {both_miss:,}  ({100*both_miss/n:.1f}%)")

    print("\n=== 判定 ===")
    rescue = (fwd_hit + rc_hit) / n
    if rescue > 0.5:
        print(f"  ✅ 可救回 {rescue*100:.0f}% 无GT read，覆盖能显著回升 → 全量翻转重建GT")
    elif rescue > 0.2:
        print(f"  ◐ 能救回 {rescue*100:.0f}%，覆盖部分回升")
    else:
        print(f"  ⚠ 仅 {rescue*100:.0f}% 可救，多数是真丢失(低质量read)")

if __name__ == '__main__':
    main()