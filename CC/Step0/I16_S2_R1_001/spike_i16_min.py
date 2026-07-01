import collections
GT_PATH="File1_ODNA.txt"; FQ_PATH="I16_S2_R1_001.fastq"; SAMPLE_RD=50000
COMP=str.maketrans("ACGTN","TGCAN")
def rc(s): return s.translate(COMP)[::-1]

gts=[l.strip() for l in open(GT_PATH) if l.strip()]
L=len(gts[0])
band=""
for pos in range(L):
    c=collections.Counter(g[pos] for g in gts if len(g)>pos)
    cons=max(c.values())/sum(c.values())
    band += "H" if cons>0.8 else ("." if cons>0.4 else "x")
print("[GT结构带] (H=保守primer, x=高熵payload/index, .=中等)")
print("  位置: "+ "".join(str(i%10) for i in range(L)))
print("  保守: "+ band)
mtf="CAACCCTT"; print(f"[GT] motif '{mtf}' 在第 {gts[0].find(mtf)} 位")

reads=[]
with open(FQ_PATH) as f:
    while len(reads)<SAMPLE_RD:
        h=f.readline()
        if not h: break
        s=f.readline().strip(); f.readline(); f.readline()
        if 40<=len(s)<=90: reads.append(s)

fwd=sum(1 for r in reads if mtf in r); rch=sum(1 for r in reads if mtf in rc(r))
use_rc=rch>fwd
print(f"[方向] 正向含motif={fwd}  反向互补含={rch}  => {'需RC反向互补' if use_rc else '保持正向'}")

proc=[rc(r) if use_rc else r for r in reads]
pc=collections.Counter(r.find(mtf) for r in proc if mtf in r)
top=pc.most_common(3)
gtpos=gts[0].find(mtf)
print(f"[结构] motif在read中位置 top3={top}  (GT中在第{gtpos}位)")
if top:
    adapter = top[0][0]-gtpos
    print(f"[推断] read头部adapter长度 ≈ {adapter}")
print("[adapter] read前12bp高频前缀 top5:")
for p,n in collections.Counter(r[:12] for r in proc).most_common(5):
    print(f"   {p}: {n}")
