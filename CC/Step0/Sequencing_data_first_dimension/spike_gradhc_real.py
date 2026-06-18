#!/usr/bin/env python3
"""
spike_gradhc_real.py
====================
在服务器上用真实 output.txt 的前 N 条 read 验证 GradHC pipeline:
  跑通 + 解析正确 + read 总数守恒 + 回查命中 + 无 rep 泄漏

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension
    python spike_gradhc_real.py                  # 默认前 5000 条
    python spike_gradhc_real.py --n 10000
    python spike_gradhc_real.py --n 5000 --max_reads_per_tag 10
"""
import os, sys, glob, subprocess, random, argparse
from collections import defaultdict, Counter

BASE_DIR   = '/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension'
GRADHC_DIR = '/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/GradHC'
REF_LEN    = 196
LEN_MIN, LEN_MAX = REF_LEN-5, REF_LEN+5
SEED       = 42

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=5000, help='取 output.txt 前 N 条做 spike')
ap.add_argument('--max_reads_per_tag', type=int, default=30)
args = ap.parse_args()

SPIKE_DIR = os.path.join(BASE_DIR, 'gradhc_spike')
os.makedirs(SPIKE_DIR, exist_ok=True)

# ---- Step0: 取前 N 条 + 预处理 + 打薄 ----
tag_to_reads = defaultdict(list)
n_read = 0
with open(os.path.join(BASE_DIR,'output.txt')) as f:
    for line in f:
        if n_read >= args.n: break
        line=line.strip()
        if not line: continue
        n_read += 1
        parts=line.split('\t',1)
        if len(parts)!=2: continue
        tag,seq=parts
        if 'N' in seq.upper(): continue
        if not (LEN_MIN<=len(seq)<=LEN_MAX): continue
        tag_to_reads[tag].append((tag,seq))

rng=random.Random(SEED)
cleaned=[]
for tag in sorted(tag_to_reads, key=lambda x:int(x)):
    rs=tag_to_reads[tag]
    cleaned.extend(rng.sample(rs,args.max_reads_per_tag) if len(rs)>args.max_reads_per_tag else rs)
N_INPUT=len(cleaned)
print(f"[Step0] 读取前 {args.n} 行 → 打薄后 {N_INPUT} reads, {len(set(t for t,_ in cleaned))} tags")

# ---- Step1: 写 GradHC 输入（单块无监督 + 占位 rep）----
placeholder='A'*REF_LEN
inp=os.path.join(SPIKE_DIR,'spike_input.txt')
with open(inp,'w',newline='\n') as f:
    f.write(placeholder+'\n'); f.write('*'*29+'\n')
    for tag,read in cleaned: f.write(read+'\n')
    f.write('\n\n')
read_to_tags=defaultdict(list)
for tag,read in cleaned: read_to_tags[read].append(tag)
n_collide = N_INPUT - len(read_to_tags)
print(f"[Step1] 写入 {N_INPUT} reads, 唯一 read={len(read_to_tags)}, 撞串实例={n_collide}")

# ---- Step2: 跑 GradHC ----
results_dir=os.path.join(GRADHC_DIR,'Results'); os.makedirs(results_dir,exist_ok=True)
pat=os.path.join(results_dir, os.path.basename(inp)+"_*.clustering_results")
for old in glob.glob(pat): os.remove(old)
env=os.environ.copy(); env["PYTHONPATH"]=GRADHC_DIR+os.pathsep+env.get("PYTHONPATH","")
print("[Step2] 运行 GradHC ...")
import time; t0=time.time()
r=subprocess.run([sys.executable,"GradHC_clustering.py","-i",inp],env=env,cwd=GRADHC_DIR,
                 capture_output=True,text=True)
if r.returncode!=0:
    print("GradHC 失败 STDERR:\n", r.stderr[-3000:]); sys.exit(1)
matches=glob.glob(pat)
if not matches:
    print("未找到输出文件:", pat); print(r.stdout[-1500:]); sys.exit(1)
res=max(matches,key=os.path.getmtime)
print(f"[Step2] 完成 {time.time()-t0:.1f}s → {res}")

# ---- Step3: 解析 ----
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

n_clusters=len(clusters)
n_parsed=sum(len(c) for c in clusters)

# ---- 守恒 & 健全性检查 ----
matched=sum(1 for c in clusters for rd in c if rd in read_to_tags)
missing=n_parsed-matched
leak=sum(1 for c in clusters for rd in c if rd==placeholder)

print(f"\n[Step3] 簇数={n_clusters}, 归簇 reads={n_parsed}")
print("\n===== 守恒 & 健全性检查 =====")
print(f"输入 reads:     {N_INPUT}")
print(f"解析 reads:     {n_parsed}   ({'守恒 ✓' if N_INPUT==n_parsed else f'不守恒 ✗ 丢{N_INPUT-n_parsed}条'})")
print(f"可回查 tag:     {matched}")
print(f"回查失败:       {missing}   ({'全命中 ✓' if missing==0 else '✗'})")
print(f"占位 rep 泄漏:  {leak}   ({'无泄漏 ✓' if leak==0 else '✗'})")

# ---- 粗略 purity/coverage（用消耗式回查）----
pool={r:list(t) for r,t in read_to_tags.items()}
pure=0; covered=set()
for c in clusters:
    tags=[]
    for rd in c:
        cand=pool.get(rd)
        if cand: tags.append(cand.pop())
    if not tags: continue
    mt,mc=Counter(tags).most_common(1)[0]
    pure+=mc; covered.add(mt)
print(f"\n粗略 Purity:    {pure/max(n_parsed,1)*100:.2f}%")
print(f"覆盖 GT tag:    {len(covered)}")
sizes=sorted((len(c) for c in clusters),reverse=True)
if sizes: print(f"簇大小:         max={sizes[0]}, med={sizes[len(sizes)//2]}, min={sizes[-1]}")
print("\n✅ Spike 通过则可放心全量跑 pipeline_seq1d_gradhc.py")