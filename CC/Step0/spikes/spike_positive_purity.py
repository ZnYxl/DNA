#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spike: 正样本纯度验证 (Positive-Pair Purity)
============================================
目的: 在为 v21 改对比学习监督信号之前, 先验证
      "用 5-mer Jaccard > theta 定义正样本对" 选出来的对里,
      真同 GT 的比例 (P_pos) 有多高.

判读:
  - 存在 theta 甜点 (P_pos>=95% 且 avg_pos>=2) 且 > Clover 簇纯度 -> 序列空间监督可行
  - 纯度够但 avg_pos<2                                          -> 需放宽/混合监督
  - 不存在甜点                                                  -> 序列空间撑不起正样本

设计要点 (避免纯度虚高):
  - Query 集: 分层抽样 (随机选 N_GT 个 GT, 取其全部 read), 保证有足够真同源对
  - 候选池 Pool: 从全量额外随机抽 POOL_SIZE 条, 代表全局异源干扰分布
  - 每个 query 在 (query ∪ pool) 里找近邻 -> 纯度不被人为压低的干扰偏置抬高

只读, 纯 CPU, 不触碰任何训练/模型/磁盘写入(除最后存一张曲线 txt).
"""
import sys, os, time, random
sys.path.insert(0, '/mnt/st_data/liangxinyi/code')
import numpy as np
from collections import defaultdict

from models.step1_data import CloverDataLoader

# ----------------------------- 配置 -----------------------------
EXP_DIR    = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/seq_1d"
OUTPUT_TXT = "/mnt/st_data/liangxinyi/code/CC/Step0/Sequencing_data_first_dimension/output.txt"
OUT_DIR    = "/mnt/st_data/liangxinyi/code/CC/Step0/spike_purity_out"

K          = 5                  # k-mer
N_GT       = 300                # 分层: 随机选多少个 GT 当 query 来源
POOL_SIZE  = 50000              # 候选池: 全局随机抽多少条代表干扰分布
THETAS     = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
EDIT_SUBSET= 500                # edit distance 趋势对照的小子集大小
SEED       = 42
# ---------------------------------------------------------------

random.seed(SEED); np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def build_kmer_bitset(seqs, k=K):
    """把每条序列编码成 5-mer 出现集合的 bitset 行向量.
    返回 (M, n_kmers) 的 uint8 矩阵 (0/1), 以及 kmer->col 映射.
    5-mer 空间 = 4^5 = 1024, 用固定索引, 不需要字典查找加速."""
    # A C G T -> 0 1 2 3
    base = {'A':0,'C':1,'G':2,'T':3}
    n_kmers = 4 ** k
    M = len(seqs)
    mat = np.zeros((M, n_kmers), dtype=np.uint8)
    for i, s in enumerate(seqs):
        s = s.upper()
        code = 0
        valid = 0
        for ch in s:
            b = base.get(ch, -1)
            if b < 0:
                code = 0; valid = 0; continue
            code = ((code << 2) | b) & (n_kmers - 1)
            valid += 1
            if valid >= k:
                mat[i, code] = 1
    return mat  # (M, 1024) 0/1


def jaccard_block(q_bits, pool_bits):
    """q_bits: (Q, D) 0/1 ; pool_bits: (P, D) 0/1
    返回 (Q, P) Jaccard 矩阵. 用整型 matmul 算交集, 行和算并集."""
    q = q_bits.astype(np.float32)
    p = pool_bits.astype(np.float32)
    inter = q @ p.T                       # (Q, P) 交集大小
    q_sz = q.sum(axis=1, keepdims=True)   # (Q,1)
    p_sz = p.sum(axis=1, keepdims=True).T # (1,P)
    union = q_sz + p_sz - inter
    union = np.maximum(union, 1e-9)
    return inter / union                  # (Q, P)


def main():
    t0 = time.time()
    print("=" * 60)
    print("📂 加载数据 + GT")
    print("=" * 60)
    dl = CloverDataLoader(EXP_DIR)
    dl.load_gt_tags(OUTPUT_TXT)
    reads = dl.reads
    gt = np.array(dl.gt_labels)
    clover = np.array(dl.clover_labels)
    N = len(reads)
    print(f"   总 read: {N}, 有 GT: {(gt>=0).sum()}, 不同 GT: {len(set(gt[gt>=0].tolist()))}")

    # ---------- 分层抽样: query 集 ----------
    gt_to_idx = defaultdict(list)
    for i, g in enumerate(gt):
        if g >= 0:
            gt_to_idx[int(g)].append(i)
    all_gts = [g for g, idxs in gt_to_idx.items() if len(idxs) >= 2]  # 至少2条才可能有正样本对
    random.shuffle(all_gts)
    chosen_gts = all_gts[:N_GT]
    query_idx = []
    for g in chosen_gts:
        query_idx.extend(gt_to_idx[g])
    query_idx = np.array(query_idx)
    print(f"   Query: {len(chosen_gts)} 个 GT -> {len(query_idx)} 条 read")

    # ---------- 候选池: 全局随机 ----------
    pool_idx = np.random.choice(N, size=min(POOL_SIZE, N), replace=False)
    # 候选池里必须包含 query 自己 (这样 query 能在池里找到同源伙伴)
    cand_idx = np.unique(np.concatenate([query_idx, pool_idx]))
    print(f"   候选池 (query ∪ pool): {len(cand_idx)} 条")

    # ---------- 编码 bitset ----------
    print("\n🧬 编码 5-mer bitset ...")
    q_seqs = [reads[i] for i in query_idx]
    c_seqs = [reads[i] for i in cand_idx]
    q_bits = build_kmer_bitset(q_seqs)
    c_bits = build_kmer_bitset(c_seqs)
    q_gt   = gt[query_idx]
    c_gt   = gt[cand_idx]
    q_cl   = clover[query_idx]
    c_cl   = clover[cand_idx]
    # query 在候选池里的自身位置 (排除自配对)
    cand_pos = {int(ci): p for p, ci in enumerate(cand_idx)}
    q_self_pos = np.array([cand_pos[int(qi)] for qi in query_idx])
    print(f"   q_bits {q_bits.shape}, c_bits {c_bits.shape}")

    # ---------- 扫 theta: Jaccard 正样本纯度 ----------
    print("\n" + "=" * 60)
    print("🔬 扫描 theta: 序列空间(5-mer Jaccard)正样本纯度")
    print("=" * 60)
    Q = len(query_idx)
    BLK = 1000
    # 预存每个 theta 的统计
    stat_tp = {th: 0 for th in THETAS}   # 真同 GT 的对数
    stat_fp = {th: 0 for th in THETAS}   # 异 GT 的对数
    stat_npos = {th: 0 for th in THETAS} # 选中的正样本对总数 (=tp+fp)
    for s in range(0, Q, BLK):
        e = min(s + BLK, Q)
        jac = jaccard_block(q_bits[s:e], c_bits)   # (b, P)
        # 屏蔽自配对
        for r in range(e - s):
            jac[r, q_self_pos[s + r]] = -1.0
        qg = q_gt[s:e]
        for th in THETAS:
            sel = jac > th                          # (b, P) 被选为正样本
            # 每个 query 选中的候选的真 GT 是否 == query 的 GT
            for r in range(e - s):
                cols = np.where(sel[r])[0]
                if len(cols) == 0:
                    continue
                same = (c_gt[cols] == qg[r])
                stat_tp[th]  += int(same.sum())
                stat_fp[th]  += int((~same).sum())
                stat_npos[th]+= len(cols)
    print(f"\n   {'theta':>6} | {'P_pos':>7} | {'avg_pos/read':>12} | {'#pairs':>10}")
    print("   " + "-" * 46)
    curve = []
    for th in THETAS:
        npos = stat_npos[th]
        ppos = stat_tp[th] / max(npos, 1)
        avgp = npos / Q
        curve.append((th, ppos, avgp, npos))
        flag = "  <-- 甜点候选" if (ppos >= 0.95 and avgp >= 2) else ""
        print(f"   {th:>6.2f} | {ppos:>7.4f} | {avgp:>12.2f} | {npos:>10d}{flag}")

    # ---------- Clover 簇标签的正样本纯度 (对照) ----------
    print("\n" + "=" * 60)
    print("🆚 对照: Clover 簇标签定义的正样本纯度")
    print("=" * 60)
    # Clover 正样本对 = 同 Clover 簇. 在 query 集内统计 (同簇对里真同 GT 的比例)
    cl_tp = 0; cl_total = 0
    cl_to_q = defaultdict(list)
    for r in range(Q):
        cl_to_q[int(q_cl[r])].append(r)
    for cl_id, members in cl_to_q.items():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                cl_total += 1
                if q_gt[members[a]] == q_gt[members[b]]:
                    cl_tp += 1
    cl_purity = cl_tp / max(cl_total, 1)
    print(f"   Clover 簇正样本纯度: {cl_purity:.4f}  ({cl_tp}/{cl_total} 对)")

    # ---------- edit distance 趋势对照 (小子集) ----------
    print("\n" + "=" * 60)
    print("🐢 edit distance 趋势对照 (小子集, 仅看趋势)")
    print("=" * 60)
    try:
        import editdistance as _ed
        has_ed = True
    except ImportError:
        has_ed = False
        print("   ⚠️ 无 editdistance 库, 跳过 (pip install editdistance 可启用)")
    if has_ed:
        sub = np.random.choice(Q, size=min(EDIT_SUBSET, Q), replace=False)
        # 在小子集内两两, 看 edit 阈值下的纯度 (归一化 edit dist)
        ED_THS = [0.05, 0.10, 0.15]  # 归一化编辑距离阈值 (越小越严)
        ed_tp = {t:0 for t in ED_THS}; ed_np = {t:0 for t in ED_THS}
        for ii in range(len(sub)):
            for jj in range(ii+1, len(sub)):
                a, b = sub[ii], sub[jj]
                d = _ed.eval(q_seqs[a], q_seqs[b]) / max(len(q_seqs[a]), 1)
                for t in ED_THS:
                    if d < t:
                        ed_np[t] += 1
                        if q_gt[a] == q_gt[b]:
                            ed_tp[t] += 1
        print(f"   {'norm_ed<':>9} | {'P_pos':>7} | {'#pairs':>8}")
        for t in ED_THS:
            p = ed_tp[t]/max(ed_np[t],1)
            print(f"   {t:>9.2f} | {p:>7.4f} | {ed_np[t]:>8d}")

    # ---------- 存曲线 ----------
    with open(os.path.join(OUT_DIR, "purity_curve.txt"), 'w') as f:
        f.write(f"# Clover 簇正样本纯度: {cl_purity:.4f} ({cl_tp}/{cl_total})\n")
        f.write("# theta  P_pos  avg_pos_per_read  n_pairs\n")
        for th, ppos, avgp, npos in curve:
            f.write(f"{th:.2f}  {ppos:.4f}  {avgp:.2f}  {npos}\n")
    print(f"\n💾 曲线已存: {OUT_DIR}/purity_curve.txt")

    # ---------- 判读 ----------
    print("\n" + "=" * 60)
    print("📊 判读")
    print("=" * 60)
    sweet = [(th, p, a) for (th, p, a, n) in curve if p >= 0.95 and a >= 2]
    best_seq_purity = max([p for (_, p, _, _) in curve], default=0)
    if sweet:
        th, p, a = sweet[0]
        print(f"   🟢 存在甜点: theta={th}, P_pos={p:.4f}, avg_pos={a:.1f}")
        print(f"      序列空间正样本纯度 {p:.4f} vs Clover 簇 {cl_purity:.4f} "
              f"(差 {p-cl_purity:+.4f})")
        if p > cl_purity:
            print(f"   ✅ 序列空间正样本比 Clover 簇更干净 -> Design 1 (序列空间监督) 可行")
        else:
            print(f"   ⚠️ 序列空间不比 Clover 簇干净 -> 收益存疑, 需讨论")
    else:
        max_p_with_pos = [(th,p,a) for (th,p,a,n) in curve if a >= 2]
        if max_p_with_pos:
            th,p,a = max(max_p_with_pos, key=lambda x:x[1])
            print(f"   🟡 无甜点(纯度<95%). 有正样本时最高纯度: theta={th}, "
                  f"P_pos={p:.4f}, avg_pos={a:.1f}")
            print(f"      -> 需放宽阈值/混合监督, 或 Jaccard 粗筛+edit 精排")
        else:
            print(f"   🔴 纯度上去时正样本归零 -> 序列空间在该数据撑不起正样本定义")
            print(f"      -> 否决纯序列监督, 考虑其他方向")

    print(f"\n⏱️ 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()