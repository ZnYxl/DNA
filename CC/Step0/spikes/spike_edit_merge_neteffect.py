#!/usr/bin/env python3
"""
spike_edit_merge_neteffect.py —— 模拟 edit 合并对整体 SR/Recall 的净效应
========================================================================
背景: edit 合并被验证"合并的对本身零误合、零both_wrong"(spike_merge_feasibility),
      但没验证"合并后整体SR是否升"。因为Clover整体欠分割(11648簇<11714 GT),
      合并只会减簇数, 可能加重欠分割, 净效应未知。

本spike: 在Clover标签上, 用**纯consensus edit**(不用GT)模拟真实可实现的合并:
  1. 每簇算MV consensus
  2. embedding质心找候选近邻对(粗筛, 模拟真实流程)
  3. 候选对 norm_edit < θ 就合并
  4. 合并后重算, 用GT评估 SR/Recall (GT只评估不参与合并决策)
对比合并前后的 SR/Recall, 看净效应。

扫几个θ看哪个最优。

判读:
  - 合并后 SR↑ 且 Recall不降 → edit合并净正收益, 改代码跑真的
  - SR↓ 或 Recall掉 → 欠分割抵消收益, 方向要调整

需要 embedding (用R1 ckpt推理质心) + tag->ref。只读。
"""
import sys, os, glob, argparse
from collections import defaultdict, Counter
import numpy as np, edlib, torch

def ned(a,b):
    if not a or not b: return 1.0
    return edlib.align(a,b,mode="NW",task="distance")['editDistance']/max(len(a),len(b))
def mv(seqs,rl=196):
    if not seqs: return ""
    L=min(Counter(len(s) for s in seqs).most_common(1)[0][0],rl);c=[]
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

@torch.no_grad()
def infer_centroids(ckpt_path, dl, cl2r, device, max_len=201, max_reads=30):
    """用ckpt推理每个簇的embedding质心(均值)"""
    sys.path.insert(0,'/mnt/st_data/liangxinyi/code')
    from models.step1_model import Step1EvidentialModel
    from models.step1_data import seq_to_onehot
    ck=torch.load(ckpt_path,map_location=device)
    sa=ck.get('args',{}); dim=sa.get('dim',256) if isinstance(sa,dict) else 256
    mlen=sa.get('max_length',201) if isinstance(sa,dict) else 201
    m=Step1EvidentialModel(dim=dim,max_length=mlen,num_clusters=max(50,len(cl2r)),device=str(device)).to(device)
    sd=ck['model_state_dict']
    if 'length_adapter.weight' in sd:
        sh=sd['length_adapter.weight'].shape
        if sh[1]==mlen and sh[0]==mlen:
            import torch.nn as nn; m.length_adapter=nn.Linear(sh[1],sh[0]).to(device)
    m.load_state_dict(sd,strict=False); m.eval()
    cents={}
    cids=sorted(cl2r.keys())
    rng=np.random.default_rng(0)
    for ci in cids:
        rids=cl2r[ci]
        use=rids if len(rids)<=max_reads else [rids[k] for k in rng.choice(len(rids),max_reads,replace=False)]
        encs=torch.stack([seq_to_onehot(dl.reads[r],mlen) for r in use]).to(device)
        if encs.shape[1]!=mlen:
            encs=encs[:,:mlen,:] if encs.shape[1]>mlen else torch.cat([encs,torch.zeros(encs.shape[0],mlen-encs.shape[1],4,device=device)],1)
        _,pooled=m.encode_reads(encs)
        cents[ci]=pooled.mean(0).cpu()
    del m; torch.cuda.empty_cache()
    return cents

def eval_sr_recall(labels_map, cons_map, gt, tag2ref, cl2r):
    """labels_map: cid->gt多数; cons_map: cid->consensus. 算SR/Recall"""
    GT_TOTAL=11826
    covered=set(); success=set()
    for cid, cons in cons_map.items():
        gts=[int(gt[i]) for i in cl2r[cid] if gt[i]>=0]
        if not gts: continue
        maj=Counter(gts).most_common(1)[0][0]
        if maj not in tag2ref: continue
        covered.add(maj)
        if ned(cons, tag2ref[maj])==0: success.add(maj)
    return len(success)/GT_TOTAL, len(covered)/GT_TOTAL, len(cons_map)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment_dir',required=True)
    p.add_argument('--gt_tags',required=True)
    p.add_argument('--ref_fasta',required=True)
    p.add_argument('--sam',required=True)
    p.add_argument('--r1_ckpt',required=True,help='R1 step1_final_model.pth, 推理质心')
    p.add_argument('--code_dir',default='/mnt/st_data/liangxinyi/code')
    p.add_argument('--topk',type=int,default=5,help='每簇找几个embedding近邻当候选')
    p.add_argument('--max_reads',type=int,default=40)
    args=p.parse_args()
    if args.code_dir not in sys.path: sys.path.insert(0,args.code_dir)
    from models.step1_data import CloverDataLoader

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl=CloverDataLoader(args.experiment_dir)
    clover=np.array(dl.clover_labels); dl.load_gt_tags(args.gt_tags)
    gt=np.array(dl.gt_labels)
    tag2ref=build_tag_to_ref(args.sam,args.ref_fasta)
    cl2r=defaultdict(list)
    for i in range(len(dl.reads)):
        if clover[i]>=0: cl2r[int(clover[i])].append(i)
    cl2r=dict(cl2r)
    print(f"   簇数={len(cl2r)}, GT匹配={int((gt>=0).sum())}")

    # consensus 缓存
    rng=np.random.default_rng(0)
    cons={}
    for ci,rids in cl2r.items():
        use=rids if len(rids)<=args.max_reads else [rids[k] for k in rng.choice(len(rids),args.max_reads,replace=False)]
        cons[ci]=mv([dl.reads[i] for i in use])

    # 基线(不合并)
    print("\n"+"="*60); print("📊 基线 (Clover, 不合并)"); print("="*60)
    sr0,rc0,nc0=eval_sr_recall(None, cons, gt, tag2ref, cl2r)
    print(f"   SR={sr0:.4f}, Recall={rc0:.4f}, 簇数={nc0}")

    # 推理质心 + 找候选近邻
    print("\n🔮 推理簇质心 (R1 ckpt)...")
    cents=infer_centroids(args.r1_ckpt, dl, cl2r, device)
    cids=sorted(cents.keys())
    cmat=torch.stack([cents[c] for c in cids])
    cmat_n=torch.nn.functional.normalize(cmat,dim=1)
    print("   找候选近邻对...")
    # topk 近邻
    cand_pairs=set()
    chunk=2000
    for s in range(0,len(cids),chunk):
        e=min(s+chunk,len(cids))
        sim=cmat_n[s:e]@cmat_n.T
        for li in range(e-s):
            gi=s+li
            sim[li,gi]=-2
            topk=torch.topk(sim[li],args.topk).indices.tolist()
            for j in topk:
                a,b=min(gi,j),max(gi,j)
                cand_pairs.add((a,b))
    print(f"   候选对: {len(cand_pairs)}")

    # 扫θ模拟合并
    print("\n"+"="*60); print("📊 不同θ下 edit合并的净效应"); print("="*60)
    print(f"   {'θ':>6}{'合并对数':>9}{'SR':>9}{'ΔSR':>9}{'Recall':>9}{'ΔRecall':>9}{'簇数':>8}")
    for th in [0.05,0.08,0.10,0.15]:
        # union-find 合并 edit<θ 的候选对
        parent={c:c for c in cids}
        def find(x):
            while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
            return x
        nmerge=0
        for (a,b) in cand_pairs:
            ca,cb=cids[a],cids[b]
            if ned(cons[ca],cons[cb])<th:
                ra,rb=find(ca),find(cb)
                if ra!=rb: parent[ra]=rb; nmerge+=1
        # 重建合并后的簇
        merged=defaultdict(list)
        for ci,rids in cl2r.items():
            merged[find(ci)].extend(rids)
        # 重算consensus
        mcons={}; 
        for ci,rids in merged.items():
            use=rids if len(rids)<=args.max_reads else [rids[k] for k in rng.choice(len(rids),args.max_reads,replace=False)]
            mcons[ci]=mv([dl.reads[i] for i in use])
        merged=dict(merged)
        sr,rc,nc=eval_sr_recall(None,mcons,gt,tag2ref,merged)
        print(f"   {th:>6.2f}{nmerge:>9}{sr:>9.4f}{sr-sr0:>+9.4f}{rc:>9.4f}{rc-rc0:>+9.4f}{nc:>8}")

    print("\n   判读: ΔSR>0 且 ΔRecall≥0 的θ → edit合并净正收益, 用该θ改代码跑真的")
    print("        若所有θ的ΔSR≤0 → 欠分割抵消, 方向需调整")

if __name__=='__main__':
    main()