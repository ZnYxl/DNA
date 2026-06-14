#!/usr/bin/env python3
"""
spike_merge_feasibility.py —— 方向A可行性: 碎片簇能否用edit认出同源
====================================================================
方向A: 每轮合并"真正同源但被Clover切开"的碎片簇, 让簇更接近真GT。
要可行, edit必须同时满足:
  1. 同源碎片簇对 consensus edit 小 (能认出该合的)
  2. 异源簇对 consensus edit 大     (不误合不该合的)
两个分布要分得开。只测1会骗人(若异源也小, 一合就污染)。

方法: 用Clover簇 + GT, 构造两组簇对, 算它们 consensus(MV) 的归一化edit:
  A. 同源碎片对: 同一GT被切成的不同簇之间 (来自1696个被切的GT)
  B. 异源对:     不同GT的簇之间 (随机)
输出两组edit分布 + AUROC + 推荐合并阈值。

判读:
  - AUROC高(>0.95)且存在阈值θ使"同源<θ覆盖率高、异源<θ很少" → 方向A可行, edit能干净分
  - 两组分布重叠(AUROC低) → edit认不出, 方向A难做, 需换信号

只读, 复用已验证的对齐。
"""
import os, sys, glob, argparse
from collections import defaultdict, Counter
import numpy as np
import edlib

def ned(a,b):
    if not a or not b: return 1.0
    return edlib.align(a,b,mode="NW",task="distance")['editDistance']/max(len(a),len(b))

def mv_consensus(read_seqs, ref_len=196):
    if not read_seqs: return ""
    from collections import Counter as C
    L=min(C(len(s) for s in read_seqs).most_common(1)[0][0], ref_len)
    cols=[]
    for p in range(L):
        v=[s[p] for s in read_seqs if p<len(s)]
        if not v: break
        cols.append(C(v).most_common(1)[0][0])
    return ''.join(cols)

def auroc(small_is_same, same_vals, diff_vals):
    # same应该小, diff应该大 -> 用 -edit 让"大=同源"统一方向
    s=np.concatenate([-same_vals, -diff_vals])
    y=np.concatenate([np.ones(len(same_vals)), np.zeros(len(diff_vals))])
    o=np.argsort(-s); y=y[o]
    npos,nneg=y.sum(),len(y)-y.sum()
    if npos==0 or nneg==0: return 0.5
    tp=np.cumsum(y); fp=np.cumsum(1-y)
    return float(np.trapz(tp/npos, fp/nneg))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment_dir', required=True)
    p.add_argument('--gt_tags', required=True)
    p.add_argument('--code_dir', default='/mnt/st_data/liangxinyi/code')
    p.add_argument('--n_pairs', type=int, default=2000)
    p.add_argument('--ref_len', type=int, default=196)
    p.add_argument('--max_reads', type=int, default=40)
    args=p.parse_args()
    if args.code_dir not in sys.path: sys.path.insert(0,args.code_dir)
    from models.step1_data import CloverDataLoader

    print("="*60); print("📂 加载 Clover标签 + GT"); print("="*60)
    dl=CloverDataLoader(args.experiment_dir)
    clover=np.array(dl.clover_labels)
    dl.load_gt_tags(args.gt_tags)
    gt=np.array(dl.gt_labels)
    reads=dl.reads
    rng=np.random.default_rng(42)

    # 簇->read, GT->簇集合
    cl_to_reads=defaultdict(list)
    gt_to_clusters=defaultdict(set)
    for i in range(len(reads)):
        if clover[i]>=0 and gt[i]>=0:
            cl_to_reads[int(clover[i])].append(i)
            gt_to_clusters[int(gt[i])].add(int(clover[i]))

    # consensus 缓存 (按需算)
    cons_cache={}
    def get_cons(cid):
        if cid not in cons_cache:
            rids=cl_to_reads[cid]
            use=rids if len(rids)<=args.max_reads else [rids[k] for k in rng.choice(len(rids),args.max_reads,replace=False)]
            cons_cache[cid]=mv_consensus([reads[i] for i in use], args.ref_len)
        return cons_cache[cid]

    # 被切成多簇的GT (碎片GT)
    frag_gts=[g for g,cs in gt_to_clusters.items() if len(cs)>=2]
    print(f"   被切成多簇的GT: {len(frag_gts)}")

    # A组: 同源碎片对
    print("\n🔬 构造同源碎片簇对 + 算 edit ...")
    same_vals=[]
    attempts=0
    while len(same_vals)<args.n_pairs and attempts<args.n_pairs*20:
        attempts+=1
        g=frag_gts[rng.integers(len(frag_gts))]
        cs=list(gt_to_clusters[g])
        if len(cs)<2: continue
        c1,c2=rng.choice(cs,size=2,replace=False)
        same_vals.append(ned(get_cons(int(c1)),get_cons(int(c2))))
    same_vals=np.array(same_vals)

    # B组: 异源簇对 (不同GT的主簇)
    print("🔬 构造异源簇对 + 算 edit ...")
    # 每个GT取它最大的簇当代表
    gt_main_cluster={}
    for g,cs in gt_to_clusters.items():
        gt_main_cluster[g]=max(cs, key=lambda c: len(cl_to_reads[c]))
    all_gts=list(gt_main_cluster.keys())
    diff_vals=[]
    while len(diff_vals)<args.n_pairs:
        g1,g2=rng.choice(all_gts,size=2,replace=False)
        diff_vals.append(ned(get_cons(gt_main_cluster[g1]),get_cons(gt_main_cluster[g2])))
    diff_vals=np.array(diff_vals)

    # 分析
    print("\n"+"="*60); print("📊 结果"); print("="*60)
    print(f"   同源碎片对 edit: n={len(same_vals)}, "
          f"median={np.median(same_vals):.4f}, mean={same_vals.mean():.4f}, "
          f"P90={np.quantile(same_vals,0.9):.4f}")
    print(f"   异源簇对   edit: n={len(diff_vals)}, "
          f"median={np.median(diff_vals):.4f}, mean={diff_vals.mean():.4f}, "
          f"P10={np.quantile(diff_vals,0.1):.4f}")
    au=auroc(True, same_vals, diff_vals)
    print(f"   AUROC (区分同源/异源): {au:.4f}")

    # 扫阈值: 不同θ下 同源召回 vs 异源误合率
    print(f"\n   {'θ(edit<)':>10}{'同源召回':>10}{'异源误合':>10}")
    for th in [0.02,0.05,0.08,0.10,0.15,0.20]:
        same_caught=(same_vals<th).mean()
        diff_wrong=(diff_vals<th).mean()
        flag=" ✓好阈值" if (same_caught>0.7 and diff_wrong<0.02) else ""
        print(f"   {th:>10.2f}{same_caught:>10.1%}{diff_wrong:>10.1%}{flag}")

    print("\n   判读:")
    print("   - 存在θ使 同源召回>70% 且 异源误合<2% → 方向A可行, edit能干净合并碎片")
    print("   - 找不到这样的θ → 同源异源edit重叠, edit认不出, 方向A需改信号")

if __name__=='__main__':
    main()