#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_gt_labels.py  ——  I16 ground-truth 标签构造 + 按簇采样

做的事:
  对 i16_clean.fasta 的每条 read, 用 edit distance 归到最近的 GT(16,383 条之一),
  这就是 GradHC 论文里真实数据集 perfect clustering 的标准构造法
  ("each read -> original strand with best edit-distance score")。
  同时按 GT 簇限制每簇最多 SAMPLE_PER_CLUSTER 条 copies(对齐论文 dataset VI 规模)。

GT 铁律: 以序列为 key, 不靠行号。edlib HW(infix) 模式处理 deletion 拉短的 read。

加速:
  - 倒排索引 (7-mer -> GT列表), read 只查表累加候选, 不遍历全部 GT
  - 屏蔽超高频 k-mer (index 固定前缀 ACAAC 造成的, 挂 >5% GT 的 k-mer 不用于预筛)
  - 多进程: 倒排索引每个 worker 各建一份(只读), reads 分块并行

输入:  File1_ODNA.txt      (16,383 条 GT, 每行一条 60nt)
       i16_clean.fasta     (脚本1输出)
输出:  i16_labeled_sampled.fasta   (已采样的 reads, id 形如 >r123|gt=4567|ed=11)
       i16_gt_labels.txt           (read_seq \t gt_id \t ed, 已采样的)
       i16_cluster_sizes.txt       (gt_id \t 采样后簇大小)

依赖: edlib  (pip install edlib --break-system-packages)
"""
import sys, os, collections, time
from multiprocessing import Pool

# ============ 配置 ============
GT_PATH    = "File1_ODNA.txt"
READ_FASTA = "i16_clean.fasta"
OUT_FASTA  = "i16_labeled_sampled.fasta"
OUT_LABELS = "i16_gt_labels.txt"
OUT_SIZES  = "i16_cluster_sizes.txt"

K          = 7          # k-mer 长度(spike 验证: 中位每kmer挂47条GT, 区分度足够)
TOPN       = 25         # 预筛候选数
NORM_ED_TH = 0.40       # 归一化ED阈值(ED/readlen), 超过判 -1 不归属
SAMPLE_PER_CLUSTER = 30 # 每个GT簇采样上限(对齐 GradHC dataset VI 的 0-32 copies)
N_PROC     = 16         # 进程数(按 kunyu 核数调; 单进程约 11ms/read)
CHUNK      = 20000      # 每个任务块的 read 数
HIGHFREQ_FRAC = 0.05    # 挂 GT 数 > 总GT*此比例 的 k-mer 屏蔽
# =============================

# ---- 全局(worker 内初始化一次) ----
_GTS = None
_INV = None
_BANNED = None


def _kmers(s, k=K):
    return set(s[i:i+k] for i in range(len(s) - k + 1))


def _init_worker(gt_path):
    """每个 worker 启动时建一次倒排索引(只读, 不跨进程共享内存但够快)。"""
    global _GTS, _INV, _BANNED
    import edlib  # noqa: 确保子进程能 import
    gts = [l.strip() for l in open(gt_path) if l.strip()]
    inv = collections.defaultdict(list)
    for i, g in enumerate(gts):
        for km in _kmers(g):
            inv[km].append(i)
    # 屏蔽超高频 k-mer(index 固定前缀造成)
    thr = max(2, int(len(gts) * HIGHFREQ_FRAC))
    banned = set(km for km, lst in inv.items() if len(lst) > thr)
    _GTS, _INV, _BANNED = gts, inv, banned


def _nearest(read):
    """返回 (gt_id, ed)。无候选或超阈值返回 (-1, ed)。"""
    import edlib
    cnt = collections.Counter()
    for km in _kmers(read):
        if km in _BANNED:
            continue
        for gid in _INV.get(km, ()):
            cnt[gid] += 1
    if not cnt:
        return -1, 999
    cand = [g for g, _ in cnt.most_common(TOPN)]
    best_id, best_ed = -1, 10**9
    for gid in cand:
        ed = edlib.align(read, _GTS[gid], mode="HW", task="distance")["editDistance"]
        if ed < best_ed:
            best_ed, best_id = ed, gid
    if len(read) == 0 or best_ed / len(read) > NORM_ED_TH:
        return -1, best_ed
    return best_id, best_ed


def _process_chunk(reads):
    """worker: 处理一批 (rid, seq), 返回 [(rid, seq, gt_id, ed), ...]。"""
    out = []
    for rid, seq in reads:
        gid, ed = _nearest(seq)
        out.append((rid, seq, gid, ed))
    return out


def read_fasta_chunks(path, chunk):
    """流式产出 [(rid, seq), ...] 块。"""
    buf = []
    rid = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                rid = line[1:]
            elif line and rid is not None:
                buf.append((rid, line))
                rid = None
                if len(buf) >= chunk:
                    yield buf
                    buf = []
    if buf:
        yield buf


def main():
    for p in (GT_PATH, READ_FASTA):
        if not os.path.exists(p):
            sys.exit(f"[ERR] 找不到 {p}")

    gts_count = sum(1 for l in open(GT_PATH) if l.strip())
    print(f"[GT] {gts_count} 条")
    print(f"[配置] k={K} topN={TOPN} 归一化ED阈值={NORM_ED_TH} "
          f"每簇采样={SAMPLE_PER_CLUSTER} 进程={N_PROC}")

    cluster_count = collections.Counter()   # gt_id -> 已采样数
    n_total = n_assigned = n_unassigned = n_sampled = 0
    t0 = time.time()

    fout = open(OUT_FASTA, "w")
    flab = open(OUT_LABELS, "w")

    with Pool(N_PROC, initializer=_init_worker, initargs=(GT_PATH,)) as pool:
        # imap 保证流式: 一块块喂, 一块块回, 内存可控
        for result in pool.imap(_process_chunk, read_fasta_chunks(READ_FASTA, CHUNK)):
            for rid, seq, gid, ed in result:
                n_total += 1
                if gid == -1:
                    n_unassigned += 1
                    continue
                n_assigned += 1
                # 边读边采: 该簇没满才收
                if cluster_count[gid] < SAMPLE_PER_CLUSTER:
                    cluster_count[gid] += 1
                    n_sampled += 1
                    fout.write(f">{rid}|gt={gid}|ed={ed}\n{seq}\n")
                    flab.write(f"{seq}\t{gid}\t{ed}\n")
            if n_total % 200000 < CHUNK:
                rate = n_total / max(1e-9, time.time() - t0)
                print(f"  ...扫描 {n_total:,}  采样 {n_sampled:,}  "
                      f"已满簇 {sum(1 for v in cluster_count.values() if v>=SAMPLE_PER_CLUSTER):,}  "
                      f"({rate:.0f} reads/s)", flush=True)

    fout.close()
    flab.close()

    # 簇大小输出
    with open(OUT_SIZES, "w") as fs:
        for gid in range(gts_count):
            fs.write(f"{gid}\t{cluster_count.get(gid,0)}\n")

    filled = sum(1 for v in cluster_count.values() if v >= SAMPLE_PER_CLUSTER)
    covered = len(cluster_count)
    print("\n========== GT 标签 + 采样完成 ==========")
    print(f"扫描 reads        : {n_total:,}")
    print(f"成功归属          : {n_assigned:,}  ({n_assigned/max(1,n_total):.1%})")
    print(f"无法归属(-1)      : {n_unassigned:,}  ({n_unassigned/max(1,n_total):.1%})")
    print(f"采样输出 reads    : {n_sampled:,}")
    print(f"覆盖到的 GT 簇    : {covered:,} / {gts_count:,}  ({covered/gts_count:.1%})")
    print(f"采满({SAMPLE_PER_CLUSTER}条)的簇  : {filled:,}")
    print(f"耗时              : {(time.time()-t0)/60:.1f} 分钟")
    print(f"\n输出:")
    print(f"  {OUT_FASTA}   (聚类输入: Clover/GradHC 共用)")
    print(f"  {OUT_LABELS}  (GT标签: 评估用, read_seq\\tgt_id\\ted)")
    print(f"  {OUT_SIZES}   (各簇采样后大小)")


if __name__ == "__main__":
    main()