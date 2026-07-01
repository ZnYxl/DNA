import edlib, random, os
GT_PATH="File1_ODNA.txt"; FQ_PATH="I16_S2_R1_001.fastq"
N_PROBE=120
gts=[l.strip() for l in open(GT_PATH) if l.strip()]

fsize=os.path.getsize(FQ_PATH)
reads=[]
with open(FQ_PATH) as f:
    while len(reads)<N_PROBE:
        f.seek(random.randint(0,fsize-400)); f.readline()
        h=f.readline()
        while h and not h.startswith("@NS"): h=f.readline()
        if not h: continue
        s=f.readline().strip()
        if 40<=len(s)<=90: reads.append(s)

# 对每条原始read(不切任何东西), 用HW模式找最近GT, 并拿到比对位置
print("orig_len | bestED | GT中匹配区间 | read前15bp")
good=bad=0
for r in reads[:40]:
    res=min((edlib.align(r,g,mode="HW",task="locations") for g in gts),
            key=lambda x:x["editDistance"])
    ed=res["editDistance"]; loc=res["locations"]
    tag="GOOD" if ed<=8 else "bad "
    if ed<=8: good+=1
    else: bad+=1
    print(f"{len(r):3d}  ED={ed:2d}  GTloc={loc}  {tag}  {r[:15]}")
print(f"\nGOOD(ED<=8)={good}  bad={bad}")

# 关键: 看bad read 是不是开头多了一段。用GOOD read的GT匹配起点 vs read长度推断
