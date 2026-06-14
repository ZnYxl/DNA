#!/usr/bin/env python3
"""
spike_simulate_no_zone3.py —— 模拟"不做Zone III隔离"的Recall/SR上界
=====================================================================
目的: 在跑几小时完整三轮之前, 先在R3现有数据上模拟"如果Zone III不隔离、
      -1 read恢复原Clover簇"会怎样, 预判Recall能回升多少。

这是乐观上界: 假设其他一切不变(encoder/簇结构), 只把-1恢复成Clover原簇,
重算consensus和Recall/SR。真跑因连锁反应可能没这么好, 但方向可预判。

关键诚实点: 恢复的簇consensus用"含被判高噪声read"算。所以同时报:
  - Recall (覆盖了多少GT) —— 恢复后必然回升(簇复活)
  - SR (consensus真对的GT数) —— 若脏read污染consensus, SR可能不回升
  对比这两个, 才知道是"真覆盖"还是"假覆盖"。

输出三组对比:
  A. 现状R3 (带-1, 评估读的轨道)
  B. 模拟: -1全恢复Clover原簇
  C. (参考)现状R1

依赖: edlib
"""
import os, sys, glob, re, argparse
from collections import defaultdict, Counter
import numpy as np
import edlib

def ned0(a, b):  # 归一化edit, 用于SR判定(==0为完美)
    if not a or not b: return 1.0
    return edlib.align(a, b, mode="NW", task="distance")['editDistance']

def majority_vote_consensus(read_seqs, ref_len=196):
    """简单majority vote生成consensus (不跑模型, 纯统计, 和MV轨道一致)"""
    if not read_seqs: return ""
    # 按位投票, 长度取众数
    from collections import Counter as C
    L = C(len(s) for s in read_seqs).most_common(1)[0][0]
    L = min(L, ref_len) if ref_len else L
    cols = []
    for p in range(L):
        votes = [s[p] for s in read_seqs if p < len(s)]
        if not votes: break
        cols.append(C(votes).most_common(1)[0][0])
    return ''.join(cols)

# ---- 对齐工具 (复用已验证的) ----
def read_fasta_ordered(path):
    pairs=[]; cur=None
    for line in open(path):
        line=line.strip()
        if not line: continue
        if line.startswith('>'): cur=int(line[1:].split()[0])
        elif cur is not None: pairs.append((cur,line.upper())); cur=None
    pairs.sort(); return [s for _,s in pairs]

def build_tag_to_ref(sam, ref_fasta):
    sns=[]
    for line in open(sam):
        if line.startswith('@SQ'):
            for f in line.split('\t'):
                if f.startswith('SN:'): sns.append(int(f[3:])); break
        elif not line.startswith('@'): break
    sns.sort(); refs=read_fasta_ordered(ref_fasta)
    n=min(len(sns),len(refs)); return {sns[k]:refs[k] for k in range(n)}

def eval_labels_scheme(labels, reads, gt, tag_to_ref, name, ref_len=196):
    """给定一套labels, 算 covered GT 数 和 SR (consensus==ref)。
       consensus用MV生成。返回(n_clusters, recall, sr)。"""
    cl_to_reads = defaultdict(list)
    for i,l in enumerate(labels):
        if l>=0: cl_to_reads[int(l)].append(i)
    GT_TOTAL = 11826
    covered_gt = set()
    success_gt = set()
    for cid, ridxs in cl_to_reads.items():
        gts=[int(gt[i]) for i in ridxs if gt[i]>=0]
        if not gts: continue
        maj=Counter(gts).most_common(1)[0][0]
        if maj not in tag_to_ref: continue
        covered_gt.add(maj)
        # MV consensus (限每簇40条加速)
        use = ridxs if len(ridxs)<=40 else [ridxs[k] for k in np.random.default_rng(0).choice(len(ridxs),40,replace=False)]
        cons = majority_vote_consensus([reads[i] for i in use], ref_len)
        if ned0(cons, tag_to_ref[maj])==0:
            success_gt.add(maj)
    recall=len(covered_gt)/GT_TOTAL
    sr=len(success_gt)/GT_TOTAL
    print(f"   [{name}] 簇数={len(cl_to_reads)}, covered_GT={len(covered_gt)}, "
          f"Recall={recall:.4f}, SR={sr:.4f} (success_GT={len(success_gt)})")
    return len(cl_to_reads), recall, sr


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment_dir', required=True)
    p.add_argument('--gt_tags', required=True)
    p.add_argument('--ref_fasta', required=True)
    p.add_argument('--sam', required=True)
    p.add_argument('--code_dir', default='/mnt/st_data/liangxinyi/code')
    p.add_argument('--r3_labels', required=True, help='R3 refined_labels.txt')
    p.add_argument('--r1_labels', required=True, help='R1 refined_labels.txt (参考)')
    args=p.parse_args()
    if args.code_dir not in sys.path: sys.path.insert(0,args.code_dir)
    from models.step1_data import CloverDataLoader

    print("="*60); print("📂 加载"); print("="*60)
    dl0 = CloverDataLoader(args.experiment_dir)        # 原始Clover
    clover0 = np.array(dl0.clover_labels)
    dl0.load_gt_tags(args.gt_tags)
    gt = np.array(dl0.gt_labels)
    reads = dl0.reads
    tag_to_ref = build_tag_to_ref(args.sam, args.ref_fasta)
    r3 = np.loadtxt(args.r3_labels, dtype=int)
    r1 = np.loadtxt(args.r1_labels, dtype=int)
    print(f"   reads={len(reads)}, gt={int((gt>=0).sum())}, tag->ref={len(tag_to_ref)}")

    print("\n" + "="*60)
    print("🔬 三组对比 (Recall = 覆盖GT, SR = consensus真对的GT)")
    print("="*60)
    # A. 现状 R3
    eval_labels_scheme(r3, reads, gt, tag_to_ref, "A 现状R3(带-1)")
    # B. 模拟: R3的-1恢复成Clover原簇
    r3_restored = r3.copy()
    noise = (r3 < 0)
    r3_restored[noise] = clover0[noise]   # -1 恢复原Clover簇
    print(f"   (模拟恢复: {int(noise.sum())} 个-1 → Clover原簇)")
    eval_labels_scheme(r3_restored, reads, gt, tag_to_ref, "B 模拟不隔离")
    # C. 参考 R1
    eval_labels_scheme(r1, reads, gt, tag_to_ref, "C 参考R1")

    print("\n" + "="*60); print("📊 判读"); print("="*60)
    print("   看 B vs A:")
    print("   - B的Recall明显>A → 不隔离能救回覆盖率 → 值得跑真的")
    print("   - 但若 B的SR 没跟着Recall涨 → 恢复的read污染了consensus(假覆盖)")
    print("     → 不隔离救了覆盖但伤了质量, 需更细策略(只恢复不太脏的)")
    print("   - B的SR ≥ A的SR 且 Recall↑ → 净赚, 直接跑真的")


if __name__=='__main__':
    main()