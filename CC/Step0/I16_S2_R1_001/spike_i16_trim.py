import edlib, random, os, collections
GT_PATH="File1_ODNA.txt"; FQ_PATH="I16_S2_R1_001.fastq"
N=200
gts=[l.strip() for l in open(GT_PATH) if l.strip()]

ILL="AGATCGGAAGAGC"   # Illumina通用接头
def trim(s):
    # a) 切3'端Illumina接头(出现即截断,允许接头本身有1-2错,用前10bp定位)
    for probe_len in (10,8,6):
        i=s.find(ILL[:probe_len])
        if i>=0: s=s[:i]; break
    # b) 切3'端 G/C 富集尾: 从尾部连续G或C(长度>=3)剥掉
    while len(s)>=3 and s[-1] in "GC" and s[-2] in "GC" and s[-3] in "GC":
        # 只剥纯GC尾,遇到混入的A/T就停
        run=0
        j=len(s)-1
        while j>=0 and s[j] in "GC": run+=1; j-=1
        if run>=3: s=s[:len(s)-run]
        else: break
        break
    return s

fsize=os.path.getsize(FQ_PATH)
raw=[]
with open(FQ_PATH) as f:
    while len(raw)<N:
        f.seek(random.randint(0,fsize-400)); f.readline()
        h=f.readline()
        while h and not h.startswith("@NS"): h=f.readline()
        if not h: continue
        s=f.readline().strip()
        if 40<=len(s)<=90: raw.append(s)

def nearest_ed(r):
    return min(edlib.align(r,g,mode="HW",task="distance")["editDistance"] for g in gts)

ed_raw=[nearest_ed(r) for r in raw]
ed_trim=[nearest_ed(trim(r)) for r in raw]
import statistics as st
print(f"[原始]   中位ED={st.median(ed_raw):.0f}  平均={st.mean(ed_raw):.1f}  ED<=8占比={sum(1 for d in ed_raw if d<=8)/N:.1%}")
print(f"[切割后] 中位ED={st.median(ed_trim):.0f}  平均={st.mean(ed_trim):.1f}  ED<=8占比={sum(1 for d in ed_trim if d<=8)/N:.1%}")
print(f"[长度] 原始中位={st.median([len(r) for r in raw]):.0f}  切割后中位={st.median([len(trim(r)) for r in raw]):.0f}")
print("\n[样例] 原始 -> 切割后:")
for r in raw[:8]:
    print(f"  {len(r):2d}|{r}")
    print(f"  {len(trim(r)):2d}|{trim(r)}  (ED {nearest_ed(r)}->{nearest_ed(trim(r))})")
