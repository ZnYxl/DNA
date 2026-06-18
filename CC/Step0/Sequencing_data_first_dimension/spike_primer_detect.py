#!/usr/bin/env python3
"""
spike_primer_detect.py
======================
探明 Seq_1D 的引物结构:不剥引物,只诊断。
对前 N 条 read 统计每个位置的碱基保守度(majority 占比),
保守度持续 >THRESH 的两端区段 = 引物候选区。

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension
    python spike_primer_detect.py
    python spike_primer_detect.py --n 8000 --thresh 0.9
"""
import argparse
from collections import Counter, defaultdict

BASE_DIR='/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension'
REF_LEN=196; LEN_MIN,LEN_MAX=191,201

ap=argparse.ArgumentParser()
ap.add_argument('--n',type=int,default=8000,help='取前 N 条对齐统计')
ap.add_argument('--thresh',type=float,default=0.85,help='保守度阈值,>此值视为引物位')
args=ap.parse_args()

reads=[]
with open(f'{BASE_DIR}/output.txt') as f:
    for line in f:
        if len(reads)>=args.n: break
        line=line.strip()
        if not line: continue
        p=line.split('\t',1)
        if len(p)!=2: continue
        seq=p[1]
        if 'N' in seq.upper(): continue
        if LEN_MIN<=len(seq)<=LEN_MAX: reads.append(seq)

print(f"统计 {len(reads)} 条 read\n")

# 左对齐保守度（前 40 位）
def conservation(seqs, positions, from_end=False):
    res=[]
    for pos in range(positions):
        c=Counter()
        for s in seqs:
            if pos<len(s):
                ch = s[-(pos+1)] if from_end else s[pos]
                c[ch.upper()]+=1
        if c:
            top,topc=c.most_common(1)[0]
            res.append((top, topc/sum(c.values())))
    return res

print("=== 左端（5'）逐位保守度 ===")
left=conservation(reads,40,from_end=False)
left_primer_end=0
for i,(b,r) in enumerate(left):
    mark='█' if r>=args.thresh else ' '
    print(f"  pos {i:2d}: {b} {r*100:5.1f}% {mark}")
    if r>=args.thresh: left_primer_end=i+1
    elif i>=3 and left_primer_end>0 and r<args.thresh:
        # 连续低于阈值则停
        if all(left[j][1]<args.thresh for j in range(i,min(i+3,len(left)))): break

print(f"\n  → 左引物候选: 0 ~ {left_primer_end} ({left_primer_end} bp)")

print("\n=== 右端（3'）逐位保守度 ===")
right=conservation(reads,40,from_end=True)
right_primer_end=0
for i,(b,r) in enumerate(right):
    mark='█' if r>=args.thresh else ' '
    print(f"  end-{i:2d}: {b} {r*100:5.1f}% {mark}")
    if r>=args.thresh: right_primer_end=i+1
    elif i>=3 and right_primer_end>0:
        if all(right[j][1]<args.thresh for j in range(i,min(i+3,len(right)))): break

print(f"\n  → 右引物候选: 末 {right_primer_end} bp")

# 提取共识引物序列
left_primer=''.join(b for b,r in left[:left_primer_end])
right_primer=''.join(b for b,r in right[:right_primer_end])[::-1]
mid_len = REF_LEN - left_primer_end - right_primer_end

print(f"\n===== 诊断结论 =====")
print(f"左引物共识({left_primer_end}bp): {left_primer}")
print(f"右引物共识({right_primer_end}bp): {right_primer}")
print(f"可变中间区: 约 {mid_len} bp  (196 - {left_primer_end} - {right_primer_end})")
print(f"\n剥引物后,Sørensen-Dice 将只比较 {mid_len}bp 可变区,跨tag误判应大幅下降")
print(f"\n建议剥离参数: --strip_left {left_primer_end} --strip_right {right_primer_end}")