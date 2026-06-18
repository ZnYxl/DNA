#!/usr/bin/env python3
"""
spike_inspect_megacluster.py
============================
直击 GradHC 巨簇:把最大簇里的 reads 回查 tag,看它到底混了多少个不同 GT tag。
不调任何参数,只诊断塌缩性质。

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension
    python spike_inspect_megacluster.py
"""
import glob, os
from collections import defaultdict, Counter

BASE_DIR='/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension'
GRADHC_DIR=BASE_DIR+'/GradHC'
TAG_MAP=BASE_DIR+'/gradhc_out/01_gradhc_tag_input.txt'  # tag\tread

# read -> tags 映射
read_to_tags=defaultdict(list)
with open(TAG_MAP) as f:
    for line in f:
        line=line.rstrip('\n')
        if not line: continue
        p=line.split('\t',1)
        if len(p)==2: read_to_tags[p[1]].append(p[0])

# 找最新结果文件
pat=os.path.join(GRADHC_DIR,'Results','01_gradhc_input.txt_*.clustering_results')
res=max(glob.glob(pat),key=os.path.getmtime)
print(f"结果文件: {res}\n")

# 解析所有簇
clusters=[]; cur=None; expect_rep=True
with open(res) as f:
    for raw in f:
        line=raw.strip()
        if line=='':
            if cur is not None and len(cur)>0: clusters.append(cur)
            cur=None; expect_rep=True; continue
        if line and line[0]=='*':
            expect_rep=False; cur=[]; continue
        if expect_rep:
            cur=None; expect_rep=True; continue
        else:
            if cur is None: cur=[]
            cur.append(line)
if cur is not None and len(cur)>0: clusters.append(cur)

clusters.sort(key=len,reverse=True)
print(f"总簇数={len(clusters)}, 最大3簇大小: {[len(c) for c in clusters[:3]]}\n")

# 消耗式回查
pool={r:list(t) for r,t in read_to_tags.items()}
print("===== Top 5 巨簇成分分析 =====")
for ci,c in enumerate(clusters[:5]):
    tags=[]
    for rd in c:
        cand=pool.get(rd)
        if cand: tags.append(cand.pop())
    if not tags:
        print(f"簇{ci}: size={len(c)}, 无法回查"); continue
    tc=Counter(tags)
    n_distinct=len(tc)
    top_tag,top_n=tc.most_common(1)[0]
    purity=top_n/len(tags)
    print(f"簇{ci}: size={len(c):>7}  含{n_distinct:>5}个不同tag  "
          f"主tag占比={purity*100:5.2f}%  (主tag={top_tag} {top_n}条)")

print("\n===== 判读 =====")
big=clusters[0]
tags=[]; pool2={r:list(t) for r,t in read_to_tags.items()}
for rd in big:
    cand=pool2.get(rd)
    if cand: tags.append(cand.pop())
n_distinct=len(set(tags))
print(f"最大簇 {len(big)} 条 read 混了 {n_distinct} 个不同 GT tag")
if n_distinct>1000:
    print("→ 极端跨tag合并:GradHC 把数千个分子揉成一团 = 真实塌缩")
    print("  这是 GradHC 在此数据+默认参数下的真实表现,可如实作为 baseline 报告")
    print("  (Clover 没塌缩 → 对比本身就是有意义的结论)")
else:
    print("→ 巨簇 tag 数不算多,可能是别的机制")