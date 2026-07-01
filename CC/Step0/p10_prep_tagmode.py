#!/usr/bin/env python3
"""
p10_prep_tagmode.py —— 为 Clover tag mode 准备全量输入（只读 output.txt，只写输入文件）
不打薄，全量过滤，统计准确唯一 tag 数。
"""
import os
from collections import Counter

OUT = '/mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009/output.txt'
DST = '/mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009/clover_tag_input.txt'
REF_LEN=200; LEN_MIN,LEN_MAX=195,205

total=n_drop=len_drop=kept=0
tags=Counter()
with open(OUT) as fin, open(DST,'w') as fout:
    for line in fin:
        line=line.rstrip('\n')
        if not line: continue
        total+=1
        p=line.split('\t',1)
        if len(p)!=2: continue
        tag,seq=p
        if 'N' in seq.upper(): n_drop+=1; continue
        if not (LEN_MIN<=len(seq)<=LEN_MAX): len_drop+=1; continue
        fout.write(f"{tag} {seq}\n")   # tag mode 要 "tag read" 空格分隔
        tags[tag]+=1; kept+=1

print(f"总行数:        {total:,}")
print(f"含N剔除:       {n_drop:,}")
print(f"长度不合格:    {len_drop:,}")
print(f"保留(写出):    {kept:,}")
print(f"唯一 tag 数:   {len(tags):,}   ← tag mode 的 -T 值")
sz=sorted(tags.values())
print(f"每tag reads:   avg={kept/len(tags):.1f} max={sz[-1]} med={sz[len(sz)//2]} min={sz[0]}")
print(f"\n输入文件: {DST}")
print(f"\n下一步 tag mode:")
print(f"  cd /mnt/st_data/liangxinyi/code/CC/Step0/Clover")
print(f"  python -m clover.main -I {DST} -L {REF_LEN} -T {len(tags)} -D 20 -V 3 -H 3")
print(f"\n  ⚠️ 跑前先确认 load_config.py 里 h_index_nums=24 e_index_nums=18")
