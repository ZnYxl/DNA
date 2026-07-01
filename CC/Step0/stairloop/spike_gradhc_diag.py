#!/usr/bin/env python3
"""
spike_gradhc_diag.py
====================
诊断 GradHC 巨簇(20万)成因：是 canonical 序列雷同，还是 MinHash(q)塌缩。

只读，5分钟。检查两件事：
  1. canonical 后序列的唯一性 —— 20万条是不是真的高度雷同
  2. GradHC numset 占比 —— MinHash 置换空间是否塌缩(对照 Seq_1D 注释的 5% 警戒线)
"""
import os, gzip, sys
from collections import Counter

BASE='/mnt/st_data/liangxinyi/code/CC/Step0/stairloop'
GRADHC_IN=os.path.join(BASE,'out_gradhc_v5','01_gradhc_input.txt')

_COMP=str.maketrans('ACGTacgt','TGCATGCA')
def revcomp(s): return s.translate(_COMP)[::-1]
def canonical(s):
    rc=revcomp(s); return s if s<=rc else rc

# 读 GradHC 输入(跳过 rep 行和 ***** 行)
seqs=[]
with open(GRADHC_IN) as f:
    for line in f:
        l=line.strip()
        if not l or l[0]=='*' or set(l)<=set('A'): continue
        seqs.append(l)
print(f"GradHC 输入序列数: {len(seqs):,}")

# 1. canonical 唯一性
uniq=set(seqs)
print(f"\n=== canonical 序列唯一性 ===")
print(f"  唯一序列数: {len(uniq):,}  ({100*len(uniq)/len(seqs):.1f}%)")
c=Counter(seqs)
top=c.most_common(5)
print(f"  出现最多的5条:")
for s,n in top:
    print(f"    {n:>6} 次  {s[:40]}...")
dup_share=sum(n for _,n in c.items() if n>1)/len(seqs)
print(f"  重复序列占比: {100*dup_share:.1f}%")

# 2. canonical 方向分布(全取正向？)
fwd=sum(1 for s in seqs if s==canonical(s) and s<=revcomp(s))
print(f"\n=== canonical 方向 ===")
n_fwd=sum(1 for s in seqs if s<=revcomp(s))
print(f"  正向被选(s<=rc): {100*n_fwd/len(seqs):.1f}%")
print(f"  (若接近100%说明A开头序列字典序总更小，canonical≈没归一化)")

# 3. MinHash 塌缩诊断：模拟 GradHC 的 numset
print(f"\n=== MinHash numset 占比 (q 塌缩诊断) ===")
sys.path.insert(0,'/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/GradHC')
try:
    import importlib
    # 直接算：q-gram 集合大小 / 4^q
    for q in [6,7,8,9,10]:
        space=4**q
        # 采样1000条算平均不同 q-gram 数
        import random; random.seed(1)
        samp=random.sample(seqs, min(1000,len(seqs)))
        avg_grams=sum(len({s[i:i+q] for i in range(len(s)-q+1)}) for s in samp)/len(samp)
        ratio=avg_grams/space
        flag="⚠塌缩" if ratio>0.05 else "✓"
        print(f"  q={q:2d}: 4^q={space:>8,}  avg_qgram={avg_grams:6.1f}  占比={ratio*100:5.2f}%  {flag}")
except Exception as e:
    print(f"  (模拟失败: {e})")

print(f"\n=== 判定 ===")
if len(uniq)/len(seqs) < 0.3:
    print("  ⚠ canonical 后序列高度雷同 → 巨簇成因是序列重复，非q")
else:
    print("  序列唯一性正常 → 巨簇成因是 MinHash q 塌缩，应增大 q")