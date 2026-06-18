#!/usr/bin/env python3
"""
spike_q_sweep.py
================
扫多个 q,为 196bp Seq_1D 选 GradHC 的最优 q。
对每个 q 报告:
  - 同源/异源 Sørensen-Dice 分布（判别力）
  - MinHash 塌缩风险:numset 平均大小 vs 置换空间 4**q 的占比
  - 默认阈值下 TP/FP

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension
    python spike_q_sweep.py
    python spike_q_sweep.py --qs 6 7 8 9
"""
import random, argparse
from collections import defaultdict
random.seed(42)

BASE_DIR='/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension'
LEN_MIN,LEN_MAX=191,201; MAX_PER_TAG=30; N_LINES=20000
BASE_VALS={"A":0,"C":1,"G":2,"T":3}

ap=argparse.ArgumentParser()
ap.add_argument('--qs',type=int,nargs='+',default=[6,7,8,9])
ap.add_argument('--n',type=int,default=N_LINES)
args=ap.parse_args()

def make_numset(seq,q):
    s=set()
    for i in range(len(seq)-q+1):
        sub=seq[i:i+q]; num=0; ok=True
        for j,ch in enumerate(sub):
            v=BASE_VALS.get(ch.upper())
            if v is None: ok=False; break
            num+=v*(4**j)
        if ok: s.add(num)
    return s
def sd(a,b):
    if not a or not b: return 0.0
    return 2*len(a&b)/(len(a)+len(b))

# 读数据
tag2reads=defaultdict(list); n=0
with open(f'{BASE_DIR}/output.txt') as f:
    for line in f:
        if n>=args.n: break
        line=line.strip()
        if not line: continue
        n+=1
        p=line.split('\t',1)
        if len(p)!=2: continue
        tag,seq=p
        if 'N' in seq.upper(): continue
        if LEN_MIN<=len(seq)<=LEN_MAX: tag2reads[tag].append(seq)
rng=random.Random(42)
tag2reads={t:(rng.sample(v,MAX_PER_TAG) if len(v)>MAX_PER_TAG else v)
           for t,v in tag2reads.items() if len(v)>=2}
tags=list(tag2reads.keys())
print(f"可用 tag={len(tags)}, reads={sum(len(v) for v in tag2reads.values())}\n")

# 固定采样的同源/异源对（跨 q 用同一批对,可比）
same_pairs=[]; 
for t in tags:
    r=tag2reads[t]
    if len(r)<2: continue
    for _ in range(min(4,len(r))):
        i,j=rng.sample(range(len(r)),2); same_pairs.append((t,i,t,j))
diff_pairs=[]
for _ in range(len(same_pairs)):
    t1,t2=rng.sample(tags,2)
    i=rng.randrange(len(tag2reads[t1])); j=rng.randrange(len(tag2reads[t2]))
    diff_pairs.append((t1,i,t2,j))

def pct(xs,q): xs=sorted(xs); return xs[min(len(xs)-1,int(q*len(xs)))]

print(f"{'q':>2} {'置换空间':>8} {'numset均值':>9} {'占比%':>7} | {'同源med':>7} {'异源med':>7} {'异源p95':>7} | {'最优阈值TP/FP':>14}")
print("-"*90)
for q in args.qs:
    top=4**q
    ns={t:[make_numset(r,q) for r in tag2reads[t]] for t in tags}
    avg_ns=sum(len(x) for t in tags for x in ns[t])/sum(len(ns[t]) for t in tags)
    same=[sd(ns[a][i],ns[c][j]) for a,i,c,j in same_pairs]
    diff=[sd(ns[a][i],ns[c][j]) for a,i,c,j in diff_pairs]
    s_med=pct(same,.5); d_med=pct(diff,.5); d_p95=pct(diff,.95)
    # 找最优阈值:TP>=98% 下 FP 最低
    best=None
    for thr in [x/100 for x in range(15,60)]:
        tp=sum(1 for x in same if x>=thr)/len(same)
        fp=sum(1 for x in diff if x>=thr)/len(diff)
        if tp>=0.98 and (best is None or fp<best[2]):
            best=(thr,tp,fp)
    bs=f"thr{best[0]:.2f}:{best[1]*100:.0f}/{best[2]*100:.1f}%" if best else "无(TP<98%)"
    flag=" ⚠塌缩风险" if avg_ns/top>0.05 else ""
    print(f"{q:>2} {top:>8} {avg_ns:>9.0f} {avg_ns/top*100:>6.1f}% | {s_med:>7.3f} {d_med:>7.3f} {d_p95:>7.3f} | {bs:>14}{flag}")

print("\n判读:")
print("  占比% = numset均值/置换空间。>5% MinHash 易碰撞塌缩(q=6 的病根)")
print("  选 q:占比低(<1%) + 同源med高 + 异源p95低 + 有阈值能 TP≥98%/FP低")
print("  GradHC 默认 final 阈值 sd_low=0.22;若最优阈值≈0.22-0.32,默认参数即可用")