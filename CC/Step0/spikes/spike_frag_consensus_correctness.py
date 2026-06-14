#!/usr/bin/env python3
"""
spike_frag_consensus_correctness.py
====================================
背景: 上一个spike发现"认不出的碎片对"(同GT但consensus edit>0.10)反而是
      较大的簇(size中位24), 排除了"小簇consensus不准"。说明同一GT被Clover
      切成的两个碎片簇, consensus已经分化(序列系统性不同)。

本spike问: 这些"认不出的碎片对", 各自consensus和真GT reference比, 谁对谁错?
  对每对(A,B), 算 A->trueGT 和 B->trueGT 的 edit:
    - A对B对: 两个都=真GT (理论上edit该小, 矛盾, 应极少)
    - 一对一错: 一个覆盖了GT, 一个没 → GT已被覆盖, 方向A不用管这部分
    - 两个都错: GT没被正确重建 → 真问题, 合并救不了

对"能合的碎片对"(edit<0.10)也做同样统计做对照。

判读:
  - 认不出对里"至少一个对"占比高 → GT其实已覆盖, 方向A只需处理能合的57%, 价值清晰
  - "两个都错"占比高 → 真问题, 需新思路(重算consensus/识别可疑簇)

需要 tag->ref 映射 (排序SN, 复用已验证对齐)。
"""
import sys, argparse
from collections import defaultdict, Counter
import numpy as np, edlib

def ned(a,b):
    if not a or not b: return 1.0
    return edlib.align(a,b,mode="NW",task="distance")['editDistance']/max(len(a),len(b))
def mv(seqs,rl=196):
    if not seqs: return ""
    L=min(Counter(len(s) for s in seqs).most_common(1)[0][0],rl); c=[]
    for p in range(L):
        v=[s[p] for s in seqs if p<len(s)]
        if not v: break
        c.append(Counter(v).most_common(1)[0][0])
    return ''.join(c)
def read_fasta_ordered(path):
    pairs=[];cur=None
    for line in open(path):
        line=line.strip()
        if not line:continue
        if line.startswith('>'):cur=int(line[1:].split()[0])
        elif cur is not None:pairs.append((cur,line.upper()));cur=None
    pairs.sort();return [s for _,s in pairs]
def build_tag_to_ref(sam,ref):
    sns=[]
    for line in open(sam):
        if line.startswith('@SQ'):
            for f in line.split('\t'):
                if f.startswith('SN:'):sns.append(int(f[3:]));break
        elif not line.startswith('@'):break
    sns.sort();refs=read_fasta_ordered(ref);n=min(len(sns),len(refs))
    return {sns[k]:refs[k] for k in range(n)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment_dir',required=True)
    p.add_argument('--gt_tags',required=True)
    p.add_argument('--ref_fasta',required=True)
    p.add_argument('--sam',required=True)
    p.add_argument('--code_dir',default='/mnt/st_data/liangxinyi/code')
    p.add_argument('--n_pairs',type=int,default=1000)
    p.add_argument('--correct_th',type=float,default=0.02,help='consensus到真GT edit<此值算"对"')
    args=p.parse_args()
    if args.code_dir not in sys.path: sys.path.insert(0,args.code_dir)
    from models.step1_data import CloverDataLoader

    dl=CloverDataLoader(args.experiment_dir)
    clover=np.array(dl.clover_labels); dl.load_gt_tags(args.gt_tags)
    gt=np.array(dl.gt_labels); reads=dl.reads
    tag2ref=build_tag_to_ref(args.sam,args.ref_fasta)
    rng=np.random.default_rng(42)

    cl2r=defaultdict(list); gt2c=defaultdict(set)
    for i in range(len(reads)):
        if clover[i]>=0 and gt[i]>=0:
            cl2r[int(clover[i])].append(i); gt2c[int(gt[i])].add(int(clover[i]))
    frag=[g for g,cs in gt2c.items() if len(cs)>=2]

    cache={}
    def cons(cid):
        if cid not in cache:
            r=cl2r[cid];u=r if len(r)<=40 else [r[k] for k in rng.choice(len(r),40,replace=False)]
            cache[cid]=mv([reads[i] for i in u])
        return cache[cid]
    # 簇->多数GT
    def cluster_gt(cid):
        gs=[int(gt[i]) for i in cl2r[cid] if gt[i]>=0]
        return Counter(gs).most_common(1)[0][0] if gs else -1

    def analyze(group_name, edit_lo, edit_hi):
        stats=Counter()
        n=0
        for _ in range(args.n_pairs*15):
            if n>=args.n_pairs: break
            g=frag[rng.integers(len(frag))]; cs=list(gt2c[g])
            if len(cs)<2: continue
            c1,c2=rng.choice(cs,2,replace=False)
            e=ned(cons(int(c1)),cons(int(c2)))
            if not (edit_lo<=e<edit_hi): continue
            n+=1
            # 各自到真GT的edit (用簇的多数GT)
            g1=cluster_gt(int(c1)); g2=cluster_gt(int(c2))
            ref1=tag2ref.get(g1); ref2=tag2ref.get(g2)
            a_ok = ref1 is not None and ned(cons(int(c1)),ref1)<args.correct_th
            b_ok = ref2 is not None and ned(cons(int(c2)),ref2)<args.correct_th
            if a_ok and b_ok: stats['both_correct']+=1
            elif a_ok or b_ok: stats['one_correct']+=1
            else: stats['both_wrong']+=1
        print(f"\n   [{group_name}] n={n}")
        for k in ['both_correct','one_correct','both_wrong']:
            print(f"      {k:>14}: {stats[k]:>5} ({stats[k]/max(n,1)*100:.1f}%)")
        return stats, n

    print("="*60); print("📊 碎片对各自consensus对真GT的正确性"); print("="*60)
    print(f"   (consensus到真GT edit<{args.correct_th}算'对')")
    analyze("认不出 edit>=0.10", 0.10, 999)
    analyze("能合   edit<0.10",  0.0,  0.10)

    print("\n   判读:")
    print("   - 认不出组'至少一个对'(one+both)占比高 → GT已被覆盖, 方向A只需处理能合的, 价值清晰")
    print("   - 认不出组'两个都错'占比高 → 真问题, 合并救不了, 需新思路")

if __name__=='__main__':
    main()