#!/usr/bin/env python3
"""
spike_sd_separability.py
========================
诊断 GradHC 塌缩真因:Sørensen-Dice 在 Seq_1D 上对「同tag vs 不同tag」是否可分。
用 GradHC 自己的 q=6 numset + sorensen_dice,采样真实 read 对统计分布。

若同源/异源分布重叠 → 数据本身难分(需换策略)
若可分但默认阈值偏低 → 调阈值即可救

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension
    python spike_sd_separability.py
"""
import random, sys, os
from collections import defaultdict
random.seed(42)

BASE_DIR='/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension'
GRADHC_DIR=BASE_DIR+'/GradHC'
sys.path.insert(0, GRADHC_DIR)
REF_LEN=196; LEN_MIN,LEN_MAX=191,201; Q=6; MAX_PER_TAG=30; N_LINES=20000

# GradHC 的 numset + sorensen_dice（直接复刻,避免 import 副作用）
BASE_VALS={"A":0,"C":1,"G":2,"T":3}
def make_numset(seq,q=Q):
    s=set()
    for i in range(len(seq)-q+1):
        sub=seq[i:i+q]
        num=0; ok=True
        for j,ch in enumerate(sub):
            v=BASE_VALS.get(ch.upper())
            if v is None: ok=False; break
            num+=v*(4**j)
        if ok: s.add(num)
    return s
def sorensen_dice(a,b):
    if not a or not b: return 0.0
    inter=len(a&b)
    return 2*inter/(len(a)+len(b))

# 读数据,按 tag 分组打薄
tag2reads=defaultdict(list)
n=0
with open(f'{BASE_DIR}/output.txt') as f:
    for line in f:
        if n>=N_LINES: break
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

# 预计算 numset
numsets={}
for t,reads in tag2reads.items():
    numsets[t]=[make_numset(r) for r in reads]

# 同源对:同 tag 内随机配对
same=[]
for t in tags:
    ns=numsets[t]
    if len(ns)<2: continue
    for _ in range(min(5,len(ns))):
        i,j=rng.sample(range(len(ns)),2)
        same.append(sorensen_dice(ns[i],ns[j]))

# 异源对:不同 tag 随机配对
diff=[]
for _ in range(len(same)):
    t1,t2=rng.sample(tags,2)
    n1=rng.choice(numsets[t1]); n2=rng.choice(numsets[t2])
    diff.append(sorensen_dice(n1,n2))

def stats(name,xs):
    xs=sorted(xs); k=len(xs)
    mean=sum(xs)/k
    p=lambda q: xs[min(k-1,int(q*k))]
    print(f"{name}: n={k}  mean={mean:.3f}  p5={p(.05):.3f}  p25={p(.25):.3f}  "
          f"med={p(.5):.3f}  p75={p(.75):.3f}  p95={p(.95):.3f}  max={xs[-1]:.3f}")
    return xs

print("===== Sørensen-Dice 分布（GradHC q=6 numset）=====")
s=stats("同源(同tag) ",same)
d=stats("异源(不同tag)",diff)

# GradHC 默认阈值
print(f"\nGradHC 默认合并阈值: chunk sd_high=0.32/sd_low=0.28, final sd_high=0.25/sd_low=0.22")
for thr in [0.22,0.25,0.28,0.32]:
    fp=sum(1 for x in d if x>=thr)/len(d)
    tp=sum(1 for x in s if x>=thr)/len(s)
    print(f"  阈值 {thr}: 同源召回(TP)={tp*100:5.1f}%  异源误并(FP)={fp*100:5.1f}%")

# 可分性结论
overlap=sum(1 for x in d if x>=min(s))/len(d)
print(f"\n异源对落入同源区间(>={min(s):.3f})的比例: {overlap*100:.1f}%")
print("\n判读:")
print("  · 若各阈值下 FP 都很高 → 数据本身 q-gram 不可分,GradHC 注定塌缩")
print("  · 若存在某阈值 TP 高且 FP 低 → 调阈值可救,我给 patch")