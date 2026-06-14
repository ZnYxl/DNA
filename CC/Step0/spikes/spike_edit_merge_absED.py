#!/usr/bin/env python3
"""用绝对ED阈值重测edit合并净效应 + 候选纯度。
关键修正: 不同GT分子可能近似重复(ED 5-15), 必须用严格绝对ED(≤2/3)而非norm<0.10。
"""
import sys
sys.path.insert(0,'/mnt/st_data/liangxinyi/code')
import numpy as np, edlib, torch
from collections import defaultdict, Counter
from models.step1_data import CloverDataLoader

EXP="/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d"
GT="/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/output.txt"
DIR="/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension"
CKPT=f"{EXP}/results/iter_1_step1/models/step1_final_model.pth"

def ed_abs(a,b):
    if not a or not b: return 999
    return edlib.align(a,b,mode="NW",task="distance")['editDistance']
def mv(seqs,rl=196):
    if not seqs: return ""
    L=min(Counter(len(s) for s in seqs).most_common(1)[0][0],rl);c=[]
    for p in range(L):
        v=[s[p] for s in seqs if p<len(s)]
        if not v:break
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

dl=CloverDataLoader(EXP); clover=np.array(dl.clover_labels)
dl.load_gt_tags(GT); gt=np.array(dl.gt_labels)
tag2ref=build_tag_to_ref(f"{DIR}/mem-se.sam", f"{DIR}/reads.fasta")
cl2r=defaultdict(list)
for i in range(len(dl.reads)):
    if clover[i]>=0: cl2r[int(clover[i])].append(i)
cl2r=dict(cl2r); rng=np.random.default_rng(0)
cons={}
for ci,rids in cl2r.items():
    use=rids if len(rids)<=40 else [rids[k] for k in rng.choice(len(rids),40,replace=False)]
    cons[ci]=mv([dl.reads[i] for i in use])
def cgt(ci):
    gs=[int(gt[i]) for i in cl2r[ci] if gt[i]>=0]
    return Counter(gs).most_common(1)[0][0] if gs else -1
GT_TOTAL=11826
def eval_sr(merged_cons, merged_map):
    cov=set();suc=set()
    for ci,c in merged_cons.items():
        gs=[int(gt[i]) for i in merged_map[ci] if gt[i]>=0]
        if not gs:continue
        maj=Counter(gs).most_common(1)[0][0]
        if maj not in tag2ref:continue
        cov.add(maj)
        if ed_abs(c,tag2ref[maj])==0:suc.add(maj)
    return len(suc)/GT_TOTAL,len(cov)/GT_TOTAL,len(merged_cons)

sr0,rc0,nc0=eval_sr(cons,cl2r)
print(f"基线: SR={sr0:.4f} Recall={rc0:.4f} 簇={nc0}")

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models.step1_model import Step1EvidentialModel
from models.step1_data import seq_to_onehot
ck=torch.load(CKPT,map_location=device);sa=ck.get('args',{})
dim=sa.get('dim',256);mlen=sa.get('max_length',201)
m=Step1EvidentialModel(dim=dim,max_length=mlen,num_clusters=max(50,len(cl2r)),device=str(device)).to(device)
sd=ck['model_state_dict']
if 'length_adapter.weight' in sd:
    sh=sd['length_adapter.weight'].shape
    if sh[1]==mlen and sh[0]==mlen:
        import torch.nn as nn;m.length_adapter=nn.Linear(sh[1],sh[0]).to(device)
m.load_state_dict(sd,strict=False);m.eval()
cids=sorted(cl2r.keys());cents={}
with torch.no_grad():
    for ci in cids:
        rids=cl2r[ci];use=rids if len(rids)<=30 else [rids[k] for k in rng.choice(len(rids),30,replace=False)]
        encs=torch.stack([seq_to_onehot(dl.reads[r],mlen) for r in use]).to(device)
        if encs.shape[1]!=mlen:encs=encs[:,:mlen,:] if encs.shape[1]>mlen else torch.cat([encs,torch.zeros(encs.shape[0],mlen-encs.shape[1],4,device=device)],1)
        _,p=m.encode_reads(encs);cents[ci]=p.mean(0).cpu()
cmat=torch.nn.functional.normalize(torch.stack([cents[c] for c in cids]),dim=1)
cand=set()
for s in range(0,len(cids),2000):
    e=min(s+2000,len(cids));sim=cmat[s:e]@cmat.T
    for li in range(e-s):
        gi=s+li;sim[li,gi]=-2
        for j in torch.topk(sim[li],5).indices.tolist():
            cand.add((min(gi,j),max(gi,j)))
print(f"候选对: {len(cand)}\n")

print(f"{'ED≤':>5}{'合并对':>8}{'同GT纯度':>9}{'SR':>9}{'ΔSR':>9}{'Recall':>9}{'ΔRecall':>9}{'簇数':>8}")
for TH in [1,2,3,5]:
    # 先算候选纯度
    pairs=[]
    for a,b in cand:
        ca,cb=cids[a],cids[b]
        if ed_abs(cons[ca],cons[cb])<=TH: pairs.append((ca,cb))
    same=sum(1 for a,b in pairs if cgt(a)==cgt(b))
    purity=same/max(len(pairs),1)
    # union-find合并
    parent={c:c for c in cids}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for ca,cb in pairs:
        ra,rb=find(ca),find(cb)
        if ra!=rb:parent[ra]=rb
    merged=defaultdict(list)
    for ci,rids in cl2r.items():merged[find(ci)].extend(rids)
    merged=dict(merged)
    mcons={}
    for ci,rids in merged.items():
        use=rids if len(rids)<=40 else [rids[k] for k in rng.choice(len(rids),40,replace=False)]
        mcons[ci]=mv([dl.reads[i] for i in use])
    sr,rc,nc=eval_sr(mcons,merged)
    print(f"{TH:>5}{len(pairs):>8}{purity:>9.3f}{sr:>9.4f}{sr-sr0:>+9.4f}{rc:>9.4f}{rc-rc0:>+9.4f}{nc:>8}")

print("\n看: 同GT纯度高(>0.95)且ΔSR≥0的ED阈值 → 安全可用")