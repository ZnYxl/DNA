#!/usr/bin/env python3
"""
spike_clover_midtree.py  —— 只读诊断，不碰生产代码
=================================================
目的: 验证 Clover 中段树 insert/检索坐标错位 bug 在 P10 上的实际影响。
方法: 复制 Clover 源码到 /tmp，monkey-patch 计数器统计命中来源分布。
输出: a_tree / b_tree / 中段树 / 新建簇 各占比；错位窗口碱基一致率。
"""
import os, sys, re, random, shutil, tempfile
from collections import Counter, defaultdict

# ───────── 配置 ─────────
SRC_CLOVER = '/mnt/st_data/liangxinyi/code/CC/Step0/Clover'
OUTPUT_TXT = '/mnt/st_data/liangxinyi/code/CC/Step0/P10_5_BDDP210000009/output.txt'
N_SAMPLE   = 50000          # 子集大小
REF_LEN    = 200
LEN_MIN, LEN_MAX = 195, 205
MAX_PER_TAG = 30
H_INDEX, E_INDEX = 24, 18
THD_LOC, FOUR_LOC = 50, 50   # 相对 payload 的中段树定位
TREE_DEPTH, V_DRIFT, H_DRIFT = 20, 3, 3
SEED = 42

print("="*60)
print("  Clover 中段树 bug 影响 spike（只读）")
print("="*60)

# ───────── 1. 读 output.txt + 长度过滤 + 打薄 ─────────
print("\n[1] 读取 + 打薄 ...")
tag_reads = defaultdict(list)
total = 0
with open(OUTPUT_TXT) as f:
    for line in f:
        line = line.rstrip('\n')
        if not line: continue
        total += 1
        p = line.split('\t', 1)
        if len(p) != 2: continue
        tag, seq = p
        if 'N' in seq.upper(): continue
        if not (LEN_MIN <= len(seq) <= LEN_MAX): continue
        tag_reads[tag].append(seq)
        if total >= 3_000_000:   # 只读前 300 万行做子集，足够
            break

rng = random.Random(SEED)
pool = []
for tag, seqs in tag_reads.items():
    s = rng.sample(seqs, MAX_PER_TAG) if len(seqs) > MAX_PER_TAG else seqs
    for q in s:
        pool.append((tag, q))
rng.shuffle(pool)
sample = pool[:N_SAMPLE]
print(f"    扫描 {total:,} 行, tag {len(tag_reads):,}, 打薄后池 {len(pool):,}, 取样 {len(sample):,}")

# ───────── 2. 复制 Clover 到 /tmp ─────────
print("\n[2] 复制 Clover 源码到临时目录 ...")
tmp = tempfile.mkdtemp(prefix='clover_spike_')
shutil.copytree(SRC_CLOVER, os.path.join(tmp, 'Clover'))
CLOVER_TMP = os.path.join(tmp, 'Clover')

# patch load_config.py 的参数
cfg = os.path.join(CLOVER_TMP, 'clover', 'load_config.py')
with open(cfg) as f: c = f.read()
for k, v in {'h_index_nums':H_INDEX,'e_index_nums':E_INDEX,
             'thd_tree_loc':THD_LOC,'four_tree_loc':FOUR_LOC,
             'read_len':REF_LEN,'end_tree_len':TREE_DEPTH,
             'Vertical_drift':V_DRIFT,'Horizontal_drift':H_DRIFT}.items():
    c = re.sub(rf'"{k}"\s*:\s*\d+', f'"{k}" : {v}', c)
with open(cfg,'w') as f: f.write(c)
print(f"    临时 Clover: {CLOVER_TMP}")

sys.path.insert(0, CLOVER_TMP)
from clover import load_config as lc
from clover import tree as tr

# ───────── 3. 复刻 cluster 逻辑 + 命中分类计数 ─────────
print("\n[3] 跑聚类(带命中分类计数器) ...")
cfgd = lc.out_put_config()
cfgd['read_len'] = REF_LEN

a_tree, b_tree = tr.Trie(), tr.Trie()
c_tree, d_tree = tr.Trie(), tr.Trie()
ref_dict = {}
fuzz_list = [THD_LOC, FOUR_LOC, cfgd['other_tree_len']]
loc_nums  = cfgd['Vertical_drift'] if isinstance(cfgd['Vertical_drift'],list) else \
            list(range(-cfgd['Vertical_drift'], cfgd['Vertical_drift']+1))
fuzz_tree_nums = H_DRIFT
tree_threshold = cfgd['tree_threshold']
now_clust_threshold = cfgd['now_clust_threshold']
dna_tree_nums = TREE_DEPTH
read_len = REF_LEN

hit = Counter()
mid_window_match = []   # 错位窗口碱基一致率采样
test_num = 0

for tag, dna_str in sample:
    test_num += 1
    dna_num = test_num
    dna_a_str = dna_str[H_INDEX:H_INDEX+dna_tree_nums]
    dna_b_str = dna_str[-E_INDEX-dna_tree_nums:-E_INDEX]

    a_align = a_tree.fuzz_fin(dna_a_str, tree_threshold)
    if a_align[1] < fuzz_tree_nums:
        ref_dict[a_align[0]].append(tag); hit['a_tree'] += 1; continue
    b_align = b_tree.fuzz_fin(dna_b_str, tree_threshold)
    if b_align[1] < fuzz_tree_nums:
        ref_dict[b_align[0]].append(tag); hit['b_tree'] += 1; continue

    # 中段树（按检索坐标 = h_index 偏移版）
    fin_align = ["", 1000]; used_i = 0
    for i in loc_nums:
        dna_c = dna_str[H_INDEX+fuzz_list[0]-i : H_INDEX+fuzz_list[0]+fuzz_list[2]-i]
        ca = c_tree.fuzz_fin(dna_c, tree_threshold)
        if ca[1] < fin_align[1]: fin_align, used_i = ca, i
        dna_d = dna_str[read_len-2-fuzz_list[1]-i-E_INDEX : read_len-2-fuzz_list[1]+fuzz_list[2]-i-E_INDEX]
        da = d_tree.fuzz_fin(dna_d, tree_threshold)
        if da[1] < fin_align[1]: fin_align, used_i = da, i

    if fin_align[1] < fuzz_tree_nums:
        ref_dict[fin_align[0]].append(tag); hit['mid_tree'] += 1
    if fin_align[1] >= now_clust_threshold:
        # 新建簇：按【源码原样】用未偏移坐标 insert（复刻 bug）
        ref_dict[dna_num] = [tag]; hit['new_cluster'] += 1
        a_tree.insert(dna_a_str, dna_num)
        b_tree.insert(dna_b_str, dna_num)
        ins_c = dna_str[fuzz_list[0]-used_i : fuzz_list[0]+fuzz_list[2]-used_i]          # 未加 h_index ← bug
        ins_d = dna_str[read_len-2-fuzz_list[1]-used_i : read_len-2-fuzz_list[1]+fuzz_list[2]-used_i]  # 未减 e_index ← bug
        c_tree.insert(ins_c, dna_num)
        d_tree.insert(ins_d, dna_num)
        # 记录: 插入窗口 vs 检索窗口实际碱基差异
        ret_c = dna_str[H_INDEX+fuzz_list[0]-used_i : H_INDEX+fuzz_list[0]+fuzz_list[2]-used_i]
        if len(ins_c)==len(ret_c)==fuzz_list[2]:
            same = sum(1 for x,y in zip(ins_c,ret_c) if x==y)
            mid_window_match.append(same/fuzz_list[2])

    if test_num % 10000 == 0:
        print(f"    {test_num:,} ...")

# ───────── 4. 报告 ─────────
print("\n" + "="*60)
print("  结果")
print("="*60)
tot = sum(hit.values())
print(f"\n  总处理: {test_num:,}")
print(f"  命中来源分布:")
for k in ['a_tree','b_tree','mid_tree','new_cluster']:
    v = hit[k]
    print(f"    {k:12s}: {v:>8,}  ({v/max(test_num,1)*100:5.2f}%)")
print(f"\n  ── bug 核心指标 ──")
print(f"  中段树命中数: {hit['mid_tree']:,}  ({hit['mid_tree']/max(test_num,1)*100:.2f}%)")
if mid_window_match:
    import statistics
    m = statistics.mean(mid_window_match)
    print(f"  插入窗口 vs 检索窗口碱基一致率: {m*100:.1f}%  (n={len(mid_window_match)})")
    print(f"    → 若接近 25%(随机): 错位严重, 中段树几乎废")
    print(f"    → 若接近 100%: 错位无害(引物区两窗口恰好相同)")
print(f"\n  解读:")
print(f"    若 mid_tree 命中占比 <1%, bug 几乎无影响, 不必改源码")
print(f"    若 mid_tree 命中占比 >5%, bug 显著, 建议修源码副本")

shutil.rmtree(tmp)
print(f"\n  已清理临时目录")
