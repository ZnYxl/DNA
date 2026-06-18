#!/usr/bin/env python3
"""从 Iterative 的 success.txt + fail.txt 提取 finalGuess, 合成 consensus_iter.fasta"""
import re, sys

OUTDIR = "/mnt/st_data/liangxinyi/code/baseline_iter/iter_out"
files = [f"{OUTDIR}/output-results-success.txt", f"{OUTDIR}/output-results-fail.txt"]
OUT   = f"{OUTDIR}/consensus_iter.fasta"
N_EXPECT = 11648

guess = {}   # cluster_num(1-based) -> finalGuess(第3行)
for fp in files:
    with open(fp) as f:
        lines = [l.rstrip("\n") for l in f]
    i = 0
    while i < len(lines):
        m = re.match(r'Cluster Num:\s*(\d+)', lines[i])
        if m:
            cnum = int(m.group(1))
            # 块: [Cluster Num] [original] [finalGuess] [Distance]
            original   = lines[i+1] if i+1 < len(lines) else ""
            finalGuess = lines[i+2] if i+2 < len(lines) else ""
            if cnum in guess:
                print(f"  ⚠️ 簇号重复: {cnum} (在 {fp})")
            guess[cnum] = finalGuess
            i += 4
        else:
            i += 1

nums = sorted(guess.keys())
print(f"提取簇数: {len(nums)}  范围: {nums[0]}~{nums[-1]}")
missing = set(range(1, N_EXPECT+1)) - set(nums)
if missing:
    print(f"  ❌ 缺失 {len(missing)} 个簇号, 例如: {sorted(missing)[:10]}")
else:
    print(f"  ✅ 1~{N_EXPECT} 完整无缺")

with open(OUT, "w") as f:
    for cnum in range(1, N_EXPECT+1):
        seq = guess.get(cnum, "")          # 缺失则空, 避免编号错位
        f.write(f">cluster_{cnum-1}\n{seq}\n")

written = sum(1 for c in range(1, N_EXPECT+1) if guess.get(c))
print(f"写入: {OUT}")
print(f"  非空 consensus: {written}/{N_EXPECT}")
