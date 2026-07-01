#!/usr/bin/env python3
"""
spike_revcomp_check.py
======================
只读验证：6% 离群 read (ED>0.10) 是否为反向互补未翻转。

逻辑：对每条 read，比正向 ED 和 反向互补 ED。
  - 若离群 read 翻转后 ED 大幅下降 → 确认 RC，应翻转救回
  - 若翻转后仍高 → 真噪声/嵌合体，应剔除

不写文件。
用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/stairloop
    python spike_revcomp_check.py
"""
import os, gzip, random
from collections import defaultdict
import edlib

BASE   = '/mnt/st_data/liangxinyi/code/CC/Step0/stairloop'
FASTQ  = os.path.join(BASE, 'sequencing reads', 'filter_sequences3.fastq')
GT_TSV = os.path.join(BASE, 'read2ref_q60.tsv')
REF    = os.path.join(BASE, 'test_encode.fasta')

SAMPLE = 5000      # 抽多少 read 来验
SEED   = 42
COMP   = str.maketrans('ACGTacgt', 'TGCATGCA')

def revcomp(s):
    return s.translate(COMP)[::-1]

def load_ref():
    d, rid = {}, None
    with open(REF) as f:
        for line in f:
            if line.startswith('>'): rid = line[1:].strip()
            elif rid: d[rid] = line.strip(); rid = None
    return d

def load_gt():
    g = {}
    with open(GT_TSV) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2: g[p[0]] = p[1]
    return g

def ned(a, b):
    return edlib.align(a, b, task='distance')['editDistance'] / len(b)

def main():
    ref, gt = load_ref(), load_gt()
    op = gzip.open(FASTQ, 'rt') if FASTQ.endswith('.gz') else open(FASTQ)

    rng = random.Random(SEED)
    n_total = n_outlier = 0
    rc_rescued = still_bad = 0
    samples = []   # (fwd_ed, rc_ed)

    with op as f:
        while True:
            h = f.readline()
            if not h: break
            seq = f.readline().strip(); f.readline(); f.readline()
            if rng.random() > SAMPLE/369455: continue
            rid = h[1:].split()[0]
            rf = gt.get(rid)
            if rf not in ref: continue
            rseq = ref[rf]
            fwd = ned(seq, rseq)
            if fwd <= 0.10:
                n_total += 1; continue
            # 离群：试反向互补
            n_total += 1; n_outlier += 1
            rc = ned(revcomp(seq), rseq)
            samples.append((round(fwd,3), round(rc,3)))
            if rc < 0.05: rc_rescued += 1
            else: still_bad += 1

    print(f"抽样 reads: {n_total:,}")
    print(f"离群 (正向ED>0.10): {n_outlier:,}  ({100*n_outlier/max(n_total,1):.1f}%)")
    if n_outlier:
        print(f"\n=== 离群 read 反向互补验证 ===")
        print(f"  翻转后 ED<0.05 (可救回): {rc_rescued:,}  ({100*rc_rescued/n_outlier:.1f}%)")
        print(f"  翻转后仍 >0.05 (真噪声): {still_bad:,}  ({100*still_bad/n_outlier:.1f}%)")
        print(f"\n  样例 (正向ED -> 反向互补ED):")
        for fwd, rc in samples[:15]:
            tag = "✅RC" if rc < 0.05 else "❌噪声"
            print(f"    {fwd:.3f} -> {rc:.3f}  {tag}")

    print("\n=== 判定 ===")
    if n_outlier and rc_rescued/n_outlier > 0.8:
        print("  ✅ 多数离群是反向互补 → 翻转救回，别丢")
    elif n_outlier and rc_rescued/n_outlier < 0.2:
        print("  ⚠ 多数翻转后仍坏 → 真噪声/嵌合体，剔除")
    else:
        print("  ◐ 混合 → 翻转能救一部分，剩余剔除")

if __name__ == '__main__':
    main()