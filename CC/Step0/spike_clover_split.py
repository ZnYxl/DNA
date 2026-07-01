#!/usr/bin/env python3
"""
spike_clover_split.py —— 排查 P10 过分裂是 spike 假象还是真实行为
不 shuffle，按 tag 聚集喂入，对比 new_cluster 占比变化。
"""
import os, sys, re, random, shutil, tempfile
from collections import Counter, defaultdict

SRC = '/mnt/st_data/liangxinyi/code/CC/Step0/Clover'
OUT = '/mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009/output.txt'
N_TAGS_SAMPLE = 2000      # 取前 2000 个 tag 的全部 read（同源相邻）
REF_LEN=200; LEN_MIN,LEN_MAX=195,205; MAX_PER_TAG=30
H_INDEX,E_INDEX=24,18; THD,FOUR=50,50
DEPTH,VD,HD=20,3,3; SEED=42

print("="*60); print("  P10 过分裂排查 spike（tag聚集 vs 打散）"); print("="*60)

# 读 + 按 tag 聚集
tag_reads=defaultdict(list); total=0
with open(OUT) as f:
    for line in f:
        line=line.rstrip('\n')
        if not line: continue
        total+=1; p=line.split('\t',1)
        if len(p)!=2: continue
        tag,seq=p
        if 'N' in seq.upper(): continue
        if not (LEN_MIN<=len(seq)<=LEN_MAX): continue
        tag_reads[tag].append(seq)
        if len(tag_reads)>=N_TAGS_SAMPLE*3 and total>=2_000_000: break

rng=random.Random(SEED)
tags_sel=list(tag_reads.keys())[:N_TAGS_SAMPLE]
# 模式A: tag聚集（同源相邻）
seq_clustered=[]
for t in tags_sel:
    s=tag_reads[t]; s=rng.sample(s,MAX_PER_TAG) if len(s)>MAX_PER_TAG else s
    for q in s: seq_clustered.append((t,q))
# 模式B: 打散
seq_shuffled=list(seq_clustered); rng.shuffle(seq_shuffled)
print(f"\n  选 {len(tags_sel)} 个 tag, 共 {len(seq_clustered):,} reads")

# 临时 Clover
tmp=tempfile.mkdtemp(prefix='clv_'); shutil.copytree(SRC,os.path.join(tmp,'C'))
CT=os.path.join(tmp,'C'); cfgp=os.path.join(CT,'clover','load_config.py')
with open(cfgp) as f: c=f.read()
for k,v in {'h_index_nums':H_INDEX,'e_index_nums':E_INDEX,'thd_tree_loc':THD,
            'four_tree_loc':FOUR,'read_len':REF_LEN,'end_tree_len':DEPTH,
            'Vertical_drift':VD,'Horizontal_drift':HD}.items():
    c=re.sub(rf'"{k}"\s*:\s*\d+',f'"{k}" : {v}',c)
with open(cfgp,'w') as f: f.write(c)
sys.path.insert(0,CT)
from clover import load_config as lc, tree as tr

def run(seqs,label):
    cfgd=lc.out_put_config(); cfgd['read_len']=REF_LEN
    aT,bT,cT,dT=tr.Trie(),tr.Trie(),tr.Trie(),tr.Trie()
    rd={}; fl=[THD,FOUR,cfgd['other_tree_len']]
    ln=cfgd['Vertical_drift']; ln=ln if isinstance(ln,list) else list(range(-VD,VD+1))
    ftn=HD; tt=cfgd['tree_threshold']; nct=cfgd['now_clust_threshold']
    dtn=DEPTH; rl=REF_LEN; hit=Counter(); tn=0
    for tag,ds in seqs:
        tn+=1; dn=tn
        a=ds[H_INDEX:H_INDEX+dtn]; b=ds[-E_INDEX-dtn:-E_INDEX]
        aa=aT.fuzz_fin(a,tt)
        if aa[1]<ftn: rd[aa[0]].append(tag); hit['a']+=1; continue
        ba=bT.fuzz_fin(b,tt)
        if ba[1]<ftn: rd[ba[0]].append(tag); hit['b']+=1; continue
        fa=["",1000]; ui=0
        for i in ln:
            dc=ds[H_INDEX+fl[0]-i:H_INDEX+fl[0]+fl[2]-i]; ca=cT.fuzz_fin(dc,tt)
            if ca[1]<fa[1]: fa,ui=ca,i
            dd=ds[rl-2-fl[1]-i-E_INDEX:rl-2-fl[1]+fl[2]-i-E_INDEX]; da=dT.fuzz_fin(dd,tt)
            if da[1]<fa[1]: fa,ui=da,i
        if fa[1]<ftn: rd[fa[0]].append(tag); hit['m']+=1
        if fa[1]>=nct:
            rd[dn]=[tag]; hit['n']+=1
            aT.insert(a,dn); bT.insert(b,dn)
            cT.insert(ds[fl[0]-ui:fl[0]+fl[2]-ui],dn)
            dT.insert(ds[rl-2-fl[1]-ui:rl-2-fl[1]+fl[2]-ui],dn)
    n_clu=len(rd); avg=tn/max(n_clu,1)
    print(f"\n  [{label}] {tn:,} reads → {n_clu:,} 簇, 平均 {avg:.2f} read/簇")
    print(f"    a={hit['a']/tn*100:.1f}% b={hit['b']/tn*100:.1f}% "
          f"mid={hit['m']/tn*100:.1f}% new={hit['n']/tn*100:.1f}%")
    return n_clu

print("\n── 模式A: tag聚集喂入（接近真实文件顺序）──")
run(seq_clustered,"聚集")
print("\n── 模式B: 完全打散（spike假象对照）──")
run(seq_shuffled,"打散")
shutil.rmtree(tmp)
print("\n  解读: 若聚集模式 new% 远低于打散模式 → 之前81%是spike假象")
print("        若两者都高(>50%) → P10 真实过分裂, 需调参/换策略")
