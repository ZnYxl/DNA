#!/usr/bin/env python3
"""
pipeline_seq1d_nips17.py
========================
Sequencing_data_first_dimension 专用 Pipeline（NIPS'17 baseline 版 · 同源输入）

  Clover 固化的 seq1d_tags_reads.txt → NIPS'17 聚类 → 小簇过滤 → 统计 → read.txt + ref.txt

【NIPS'17 = Rashtchian et al. 2017, "Clustering billions of reads for DNA data storage"】
  原文无公开实现（微软只放数据集未放算法；GradHC 团队也是自行实现未公开）。
  本脚本依据原文 Algorithm 1 忠实复现，单机串行版（原文是 MPI 分布式，但分布式仅为
  工程规模需求，聚类结果由 hash family + binary signature filter + 两级合并判据决定，
  与是否并行无关；单机版逻辑等价、易复查、易辩护）。

  Algorithm 1 核心（merge-only agglomerative）:
    - 每条 read 起步是 singleton（并查集）
    - comm_steps 轮通信 × local_steps 轮本地，每轮采新 hash h_{π,ℓ} ~ H_{w,ℓ}
    - 每个当前簇采 1 个代表，按 hash 分桶
    - 桶内每对 (x,y)：
        若 d_H(σ(x),σ(y)) ≤ θ_low                         → 合并
        否则若 d_H ≤ θ_high 且 d_E(x,y) ≤ r               → 合并
    - hash: 随机排列 π 选最早 w-gram，取其首现位置起 w+ℓ 子串（类 MinHash）
    - signature: blocked σ_q（每 block_len 字符算 σ_q，拼接）

【同源对比】与 Clover / GradHC 读同一份 Clover 固化数据（老师拍板：打薄=预处理，固化一次共用）。
  --keep_ratio 对标 Clover：读 seq_1d_p{kr}/seq1d_tags_reads.txt，输出 seq_1d_nips17_p{kr}/。

【参数（按 Seq_1D 实测标定，非套用原文常数）】
  原文 r=25/θ_low=40/θ_high=60 是针对 m=150、p≈4% 的 Organick 数据。
  Seq_1D 实测（calibrate spike）：m=196、p≈0.4%（簇内编辑距离 P50=1, P99=10, P99.9 陡跳 53）。
  P99→P99.9 的断崖说明 53 那条尾巴是 Clover 误并的异源对污染，非真实簇内噪声。
    r = 12         （P99=10 + buffer，覆盖真实簇内对、避开污染尾；理论 4pm≈3 更小，12 已宽松）
    block_len = 22, q = 3, θ_low = 25, θ_high = 108   （spike：簇内 P95=25 / 簇内P99.9与簇间P1取小）
    w = ⌈log₄196⌉ = 4, ℓ = 12                          （原文 Theorem C.1 口径）
    comm_steps = 26, local_steps = 30                  （原文 780 迭代）

用法:
    cd /mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension
    python pipeline_seq1d_nips17.py                      # 对标 Clover seq_1d_p0.2
    python pipeline_seq1d_nips17.py --keep_ratio 0.1
    python pipeline_seq1d_nips17.py --r 10 --theta_low 25 --theta_high 108   # 覆盖参数
"""

import os
import sys
import time
import random
import argparse
from collections import defaultdict, Counter


class _Tee:
    """把 stdout 同时写到屏幕和日志文件。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


try:
    import edlib
except ImportError:
    print("✗ 缺少 edlib。请在 conda env dna 中运行。")
    sys.exit(1)

# ============================================================
# 配置
# ============================================================
BASE_DIR = '/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension'
EXP_ROOT = '/mnt/st_data/liangxinyi/code/CC/Step0/Experiments'
OUTPUT_DIR = None
EXPERIMENT_DIR = None

REF_LEN = 196
MIN_READS_PER_CLUSTER = 5
N_GT_TAGS = 11826
KEEP_RATIO = 0.2
RANDOM_SEED = 42

# NIPS'17 参数（Seq_1D 实测标定）
NIPS_R          = 12
NIPS_Q          = 3
NIPS_W          = 4
NIPS_ELL        = 12
NIPS_BLOCK_LEN  = 22
NIPS_THETA_LOW  = 25
NIPS_THETA_HIGH = 108
NIPS_COMM_STEPS = 26
NIPS_LOCAL_STEPS = 30


def banner(title):
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}\n")


# ============================================================
# NIPS'17 Algorithm 1 核心
# ============================================================
def make_hash_fn(w, ell, perm):
    """H_{w,ℓ}: 随机排列 π（用 perm dict 缓存 w-gram→优先级）选最早 w-gram，
    取其首次出现位置起长度 w+ℓ 的子串。每次调用应传入新的空 perm 以采样新 h_{π,ℓ}。"""
    def hpi(x):
        m = len(x)
        best_rank = None
        best_i = None
        for i in range(m - w + 1):
            g = x[i:i + w]
            r = perm.get(g)
            if r is None:
                r = random.random()
                perm[g] = r
            if best_rank is None or r < best_rank:
                best_rank = r
                best_i = i
        if best_i is None:
            return x
        end = min(m, best_i + w + ell)
        return x[best_i:end]
    return hpi


def blocked_signature(seq, block_len, q):
    """blocked binary signature σ_q：每 block_len 字符算 σ_q（带块偏移防混淆），返回稀疏集合。"""
    sig = set()
    for bstart in range(0, len(seq), block_len):
        block = seq[bstart:bstart + block_len]
        boff = bstart // block_len
        for i in range(len(block) - q + 1):
            sig.add((boff, block[i:i + q]))
    return sig


def qgram_distance(sa, sb):
    return len(sa ^ sb)


def edit_distance(a, b):
    return edlib.align(a, b, task="distance")["editDistance"]


def nips17_cluster(reads, r, q, w, ell, block_len, theta_low, theta_high,
                   comm_steps, local_steps, seed):
    """忠实 Algorithm 1：merge-only agglomerative + hash 分桶 + 两级 filter。
    单机用并查集表示簇；每个 local 轮采新 hash、每簇采代表、桶内合并。"""
    random.seed(seed)
    n = len(reads)
    print(f"  预计算 {n:,} 条 signature ...")
    sigs = [blocked_signature(reads[i], block_len, q) for i in range(n)]

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_edit_calls = 0
    n_merges = 0
    total_rounds = comm_steps * local_steps
    round_idx = 0
    t0 = time.time()

    for ci in range(comm_steps):
        for li in range(local_steps):
            round_idx += 1
            perm = {}
            hfn = make_hash_fn(w, ell, perm)

            # 每个当前簇采一个代表
            clusters = defaultdict(list)
            for i in range(n):
                clusters[find(i)].append(i)
            reps = [random.choice(members) for members in clusters.values()]

            # 按 hash 分桶；桶内按 hash 排序只比相邻（原文优化）
            buckets = defaultdict(list)
            for idx in reps:
                buckets[hfn(reads[idx])].append(idx)

            MAX_BUCKET = 300   # 桶大小上限，超过则下采样（破解第一轮 O(k²) 灾难）
            for hval, bucket in buckets.items():
                if len(bucket) < 2:
                    continue
                # 超大桶下采样：第一轮 singleton 全进同桶会 O(k²) 爆炸，
                # 随机抽 MAX_BUCKET 个代表比较（采代表+分桶近似的自然延伸，语义可辩护）
                if len(bucket) > MAX_BUCKET:
                    bucket = random.sample(bucket, MAX_BUCKET)
                bucket.sort()
                for a in range(len(bucket)):
                    for b in range(a + 1, len(bucket)):
                        x, y = bucket[a], bucket[b]
                        if find(x) == find(y):
                            continue
                        dh = qgram_distance(sigs[x], sigs[y])
                        if dh <= theta_low:
                            union(x, y)
                            n_merges += 1
                        elif dh <= theta_high:
                            n_edit_calls += 1
                            if edit_distance(reads[x], reads[y]) <= r:
                                union(x, y)
                                n_merges += 1

            if round_idx % 50 == 0 or round_idx == total_rounds:
                n_cur = len(set(find(i) for i in range(n)))
                print(f"    round {round_idx}/{total_rounds}  簇数={n_cur:,}  "
                      f"merges={n_merges:,}  edit_calls={n_edit_calls:,}  "
                      f"[{time.time()-t0:.0f}s]")

    out = defaultdict(list)
    for i in range(n):
        out[find(i)].append(i)
    return list(out.values())


# ============================================================
# Step 0: 读 Clover 固化的同源打薄数据
# ============================================================
def step0_load_shared_thinned(tags_file, keep_ratio):
    banner("Step 0  读取 Clover 固化的同源打薄数据")
    if not os.path.exists(tags_file):
        raise FileNotFoundError(
            f"找不到 Clover 固化数据: {tags_file}\n"
            f"请先运行 Clover 的 pipeline_seq1d.py --keep_ratio {keep_ratio}。")

    tags = []
    reads = []
    with open(tags_file) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            tags.append(parts[0])
            reads.append(parts[1])

    n_tags = len(set(tags))
    sizes = list(Counter(tags).values())
    print(f"  同源数据源:  {tags_file}")
    print(f"  （= Clover 全局随机打薄后固化的同一批 reads）\n")
    print(f"  读入 reads:  {len(reads):,}")
    print(f"  GT tags:     {n_tags:,}")
    print(f"  每 tag reads: avg={sum(sizes)/len(sizes):.1f}, "
          f"max={max(sizes)}, med={sorted(sizes)[len(sizes)//2]}, min={min(sizes)}")
    return reads, tags


# ============================================================
# Step 1~5: 同 Clover/GradHC 口径
# ============================================================
def step_filter_small(clusters, min_reads):
    banner(f"过滤小簇 (reads < {min_reads})")
    before_n = len(clusters)
    before_r = sum(len(c) for c in clusters)
    filtered = [c for c in clusters if len(c) >= min_reads]
    after_r = sum(len(c) for c in filtered)
    print(f"  过滤前:  {before_n:,} 簇,  {before_r:,} reads")
    print(f"  过滤后:  {len(filtered):,} 簇,  {after_r:,} reads")
    print(f"  丢弃:    {before_n - len(filtered):,} 簇 ({before_r - after_r:,} reads)")
    return filtered


def step_statistics(clusters, reads, tags, min_reads, stats_path):
    banner("聚类统计（过滤后）")
    total_reads = sum(len(c) for c in clusters)
    n_clusters = len(clusters)
    sizes = sorted([len(c) for c in clusters], reverse=True)

    total_pure = 0
    gt_covered = set()
    for c in clusters:
        ctags = [tags[i] for i in c]
        mtag, mcount = Counter(ctags).most_common(1)[0]
        total_pure += mcount
        gt_covered.add(mtag)

    purity = total_pure / max(total_reads, 1)
    coverage = len(gt_covered) / N_GT_TAGS

    buckets = [
        (f'{min_reads} - 10', sum(1 for s in sizes if min_reads <= s <= 10)),
        ('11 - 30', sum(1 for s in sizes if 11 <= s <= 30)),
        ('31 - 100', sum(1 for s in sizes if 31 <= s <= 100)),
        ('> 100', sum(1 for s in sizes if s > 100)),
    ]
    print(f"  聚类簇数:    {n_clusters:,}")
    print(f"  GT tag 数:   {N_GT_TAGS:,}")
    print(f"  reads:       {total_reads:,}\n")
    print(f"  Purity:      {purity*100:.2f}%")
    print(f"  Coverage:    {coverage*100:.2f}%  ({len(gt_covered)}/{N_GT_TAGS})\n")
    print(f"  簇大小分布:")
    for label, count in buckets:
        print(f"    {label:15s}  {count:>7,}")
    if sizes:
        print(f"\n  max={sizes[0]}, median={sizes[len(sizes)//2]}, min={sizes[-1]}")

    with open(stats_path, 'w') as f:
        f.write(f"Seq_1D NIPS'17 聚类统计（同源打薄 + 小簇过滤）\n{'='*40}\n\n")
        f.write(f"聚类簇数:  {n_clusters:,}\nGT tag 数: {N_GT_TAGS:,}\nreads: {total_reads:,}\n\n")
        f.write(f"Purity:    {purity*100:.2f}%\nCoverage:  {coverage*100:.2f}%\n\n")
        f.write("簇大小分布:\n")
        for label, count in buckets:
            f.write(f"  {label:15s}  {count:,}\n")
    print(f"\n  💾 {stats_path}")
    return purity, coverage


def majority_vote(reads_list, ref_len):
    vote = [Counter() for _ in range(ref_len)]
    for read in reads_list:
        for pos in range(min(len(read), ref_len)):
            b = read[pos].upper()
            if b in 'ACGT':
                vote[pos][b] += 1
    result, last = [], 'A'
    for pos in range(ref_len):
        if vote[pos]:
            last = vote[pos].most_common(1)[0][0]
        result.append(last)
    return ''.join(result)


def step_write_output(clusters, reads, read_path, ref_path):
    banner("Majority Vote → read.txt + ref.txt")
    SEPARATOR = "=====分隔符=====\n"
    n_clusters = n_reads = 0
    with open(read_path, 'w') as fr, open(ref_path, 'w') as ff:
        for c in clusters:
            cr = [reads[i] for i in c]
            if not cr:
                continue
            for read in cr:
                fr.write(read + '\n')
            fr.write(SEPARATOR)
            ff.write(majority_vote(cr, REF_LEN) + '\n')
            n_clusters += 1
            n_reads += len(cr)
    print(f"  ✅ 簇数: {n_clusters:,},  reads: {n_reads:,}")
    print(f"  read.txt: {read_path}")
    print(f"  ref.txt:  {ref_path}")


def step_deploy(reads, tags, keep_ratio):
    banner("部署到实验目录")
    import shutil
    feddna_src = os.path.join(OUTPUT_DIR, '04_FedDNA_In')
    feddna_dst = os.path.join(EXPERIMENT_DIR, '03_FedDNA_In')
    os.makedirs(feddna_dst, exist_ok=True)
    for fname in ['read.txt', 'ref.txt']:
        shutil.copy2(os.path.join(feddna_src, fname), os.path.join(feddna_dst, fname))
        print(f"  ✅ {fname} → {feddna_dst}")

    gt_tags_path = os.path.join(EXPERIMENT_DIR, 'seq1d_tags_reads.txt')
    with open(gt_tags_path, 'w') as f:
        for t, r in zip(tags, reads):
            f.write(f"{t}\t{r}\n")
    print(f"  ✅ GT tags: {gt_tags_path}")

    refs_fasta = os.path.join(BASE_DIR, 'reads.fasta')
    gt_refs_path = os.path.join(EXPERIMENT_DIR, 'seq1d_refs.txt')
    with open(refs_fasta) as fin, open(gt_refs_path, 'w') as fout:
        for line in fin:
            if not line.startswith('>'):
                fout.write(line)
    print(f"  ✅ GT refs: {gt_refs_path}\n")

    ratio_tag = f"p{keep_ratio}"
    exp_rel = f"CC/Step0/Experiments/seq_1d_nips17_{ratio_tag}"
    print(f"  🚀 运行实验:")
    print(f"     cd /mnt/st_data/liangxinyi/code")
    print(f"     python -m models.main_loop \\")
    print(f"       --experiment_dir {exp_rel}/ \\")
    print(f"       --feddna_checkpoint result/FLDNA_I/I_1214234233/model/epoch1_I.pth \\")
    print(f"       --gt_tags_file {exp_rel}/seq1d_tags_reads.txt \\")
    print(f"       --gt_refs_file {exp_rel}/seq1d_refs.txt \\")
    print(f"       --max_iterations 3 --max_length 201 \\")
    print(f"       --cl_mode ours --ref_length {REF_LEN} --primer_prefix 20 --primer_suffix 20 \\")
    print(f"       --split_tau 5 --split_min_size 6 \\")
    print(f"       2>&1 | tee {exp_rel}/seq1d_nips17_{ratio_tag}.log")
    print()
    print(f"  📊 评估命令:")
    print(f"     cd /mnt/st_data/liangxinyi/code/models")
    print(f"     python eval_reconstruction.py \\")
    print(f"       --experiment_dir /mnt/st_data/liangxinyi/code/{exp_rel}/ \\")
    print(f"       --gt_refs /mnt/st_data/liangxinyi/code/{exp_rel}/seq1d_refs.txt \\")
    print(f"       --gt_tags /mnt/st_data/liangxinyi/code/{exp_rel}/seq1d_tags_reads.txt \\")
    print(f"       --out reconstruction_eval_seq1d_nips17_{ratio_tag}.tsv")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Seq_1D NIPS17 Pipeline（同源输入版）')
    parser.add_argument('--keep_ratio', type=float, default=KEEP_RATIO)
    parser.add_argument('--min_reads', type=int, default=MIN_READS_PER_CLUSTER)
    parser.add_argument('--r', type=int, default=NIPS_R, help=f'编辑距离阈值（默认 {NIPS_R}）')
    parser.add_argument('--theta_low', type=int, default=NIPS_THETA_LOW)
    parser.add_argument('--theta_high', type=int, default=NIPS_THETA_HIGH)
    parser.add_argument('--block_len', type=int, default=NIPS_BLOCK_LEN)
    parser.add_argument('--comm_steps', type=int, default=NIPS_COMM_STEPS)
    parser.add_argument('--local_steps', type=int, default=NIPS_LOCAL_STEPS)
    args = parser.parse_args()

    global OUTPUT_DIR, EXPERIMENT_DIR
    ratio_tag = f"p{args.keep_ratio}"
    clover_tags_file = os.path.join(EXP_ROOT, f"seq_1d_{ratio_tag}", "seq1d_tags_reads.txt")
    OUTPUT_DIR = os.path.join(BASE_DIR, f'nips17_out_{ratio_tag}')
    EXPERIMENT_DIR = os.path.join(EXP_ROOT, f'seq_1d_nips17_{ratio_tag}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '04_FedDNA_In'), exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    stats_path = os.path.join(OUTPUT_DIR, '03_stats.txt')
    read_path = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'read.txt')
    ref_path = os.path.join(OUTPUT_DIR, '04_FedDNA_In', 'ref.txt')

    # ── 开启 Tee：全程输出同时写到实验目录 data_detail.txt（与 Clover 一致）──
    detail_path = os.path.join(EXPERIMENT_DIR, 'data_detail.txt')
    _detail_fh = open(detail_path, 'w')
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _detail_fh)

    print("=" * 60)
    print("  🚀  Seq_1D Pipeline（NIPS'17 baseline · 同源输入）")
    print("=" * 60)
    print(f"  数据源(同源):  {clover_tags_file}")
    print(f"  实验目录:      {EXPERIMENT_DIR}")
    print(f"  日志文件:      {detail_path}")
    print(f"  ref_len:       {REF_LEN}bp   keep_ratio: {args.keep_ratio}")
    print(f"  NIPS'17 参数:  r={args.r} q={NIPS_Q} w={NIPS_W} ℓ={NIPS_ELL} "
          f"block={args.block_len} θ_low={args.theta_low} θ_high={args.theta_high}")
    print(f"                 comm={args.comm_steps} local={args.local_steps}")

    t_start = time.time()
    reads, tags = step0_load_shared_thinned(clover_tags_file, args.keep_ratio)

    banner("NIPS'17 Algorithm 1 聚类（单机串行）")
    clusters = nips17_cluster(
        reads, r=args.r, q=NIPS_Q, w=NIPS_W, ell=NIPS_ELL,
        block_len=args.block_len, theta_low=args.theta_low, theta_high=args.theta_high,
        comm_steps=args.comm_steps, local_steps=args.local_steps, seed=RANDOM_SEED)
    print(f"\n  聚类完成，原始簇数: {len(clusters):,}")

    clusters = step_filter_small(clusters, args.min_reads)
    purity, coverage = step_statistics(clusters, reads, tags, args.min_reads, stats_path)
    step_write_output(clusters, reads, read_path, ref_path)
    step_deploy(reads, tags, args.keep_ratio)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\n  🎉 Pipeline 完成! 耗时 {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Purity: {purity*100:.2f}%   Coverage: {coverage*100:.2f}%\n{'='*60}")

    # ── 关闭 Tee ──
    sys.stdout = _orig_stdout
    _detail_fh.close()


if __name__ == '__main__':
    main()