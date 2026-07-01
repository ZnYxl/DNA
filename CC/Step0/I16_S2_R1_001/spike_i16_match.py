import edlib, collections, random
GT_PATH="File1_ODNA.txt"; FQ_PATH="I16_S2_R1_001.fastq"
N_PROBE=200          # 抽多少条read去比对
ADAP="AGATCGGAAGAGC" # Illumina adapter

gts=[l.strip() for l in open(GT_PATH) if l.strip()]
print(f"[GT] {len(gts)} 条")

def strip_adapter(s):
    i=s.find(ADAP[:8])         # 接头出现就从那截断
    return s[:i] if i>0 else s

# 随机抽 read（跳过文件不同位置，避免只取开头）
import os
fsize=os.path.getsize(FQ_PATH)
reads=[]
with open(FQ_PATH) as f:
    for _ in range(N_PROBE*4):
        f.seek(random.randint(0,fsize-400))
        f.readline()                       # 丢弃半行
        while True:
            h=f.readline()
            if not h: break
            if h.startswith("@NS"):
                s=f.readline().strip(); break
        else:
            continue
        s=strip_adapter(s)
        if 40<=len(s)<=80: reads.append(s)
        if len(reads)>=N_PROBE: break

print(f"[READ] 实际比对 {len(reads)} 条 (已切adapter)")
dists=[]
for r in reads:
    best=min(edlib.align(r,g,mode="HW",task="distance")["editDistance"] for g in gts)
    dists.append(best)

dc=collections.Counter(dists)
print("[最近GT编辑距离分布]  ED : 条数")
for d in sorted(dc):
    print(f"   ED={d:3d}: {dc[d]:4d}  {'#'*dc[d]}")
import statistics
print(f"[汇总] 中位ED={statistics.median(dists)}  平均={statistics.mean(dists):.1f}  ED<=8占比={sum(1 for d in dists if d<=8)/len(dists):.1%}")
