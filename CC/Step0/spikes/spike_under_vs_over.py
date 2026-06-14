#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spike_under_vs_over.py
======================
诊断目标(第8节): Clover 初始聚类到底是欠分割还是过分割主导,
                  "拆分混合簇"和"合并碎片簇"哪个方向能救回更多 success。

只读 spike。不碰任何 model / refined_labels / centroids / Zone 文件。
只用: read.txt (reads + Clover 初始标签) + GT (reads.fasta + output.txt)。
所以测的是 Clover 初始聚类的结构性损失,与迭代/Zone III 无关。

口径与 eval_reconstruction.py 完全一致:
  - GT 链路: load_gt_tags_file -> build_tag_to_ref_mapping -> match_reads_to_gt
  - success 判定: 真实 MV 重建出的 consensus 与真 reference 的 levenshtein == 0
  - MV 投票: 逐位多数投票 + 50% has_vote 门限 (与 compute_mv_consensus / ds_fusion_masked 同口径)

两侧损失定义
-----------
[过分割侧] 合并能救的上界:
  对每个 "被 Clover 切成多簇的 GT", 看它的所有碎片簇(以该 GT 为主GT的簇)。
  若 没有任何单个碎片 单独 MV 能 success(ED==0),
  但 把所有碎片的 read 合起来 MV 能 success
  -> 这个 GT 是 "合并可救"。

[欠分割侧] 拆分能救的上界:
  对每个混合簇(内部含 >=2 个不同 GT), 主GT = read 数最多者。
  对每个 次要GT(非主GT): 抽出它在该簇内的 read, 单独 MV,
  若 ED==0 (单独成簇能重建对自己的 reference)
  -> 这个 (簇,次要GT) 是 "拆分可救" 的一个 success。
  注意: 主GT 在现状下若已 success 不重复计; 我们只数 "拆分新增的 success"。

两个上界都是 乐观上界(假设拆/合完美执行)。实际增益 <= 上界。
对比两个数 -> 判断哪个方向天花板高。

输出
----
  - 整体结构: GT 总数 / 簇总数 / 纯簇 / 混合簇 / 被切GT数
  - 现状 baseline: 现有 Clover 聚类下多少 GT 已 success (同口径自测,用于交叉验证 SR)
  - 合并可救上界 (过分割侧)
  - 拆分可救上界 (欠分割侧)
  - 对 SR 的影响 + 对 read_util/覆盖的影响
"""
import argparse
import os
import sys
import numpy as np
from collections import defaultdict, Counter

# ── 复用 eval_reconstruction.py 的全部 GT / IO / MV 口径 ──────────────────────
# spike 在 CC/Step0/spikes/, eval_reconstruction.py 在 code/models/。
# 自动向上回溯找到 code 根(含 models/eval_reconstruction.py)并加入 sys.path,
# 这样无论 spike 放哪、import 链("from models.xxx")都能正常工作。
def _add_code_root():
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(8):  # 最多向上回溯 8 层
        cand = os.path.join(d, 'models', 'eval_reconstruction.py')
        if os.path.exists(cand):
            if d not in sys.path:
                sys.path.insert(0, d)            # code 根 -> 支持 from models.xxx
            mdir = os.path.join(d, 'models')
            if mdir not in sys.path:
                sys.path.insert(0, mdir)         # models/ -> 支持 from eval_reconstruction
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # 兜底: 也试 spike 同目录
    if here not in sys.path:
        sys.path.insert(0, here)
    return None

_code_root = _add_code_root()
if _code_root:
    print(f"[path] code 根: {_code_root}")

try:
    from eval_reconstruction import (
        levenshtein,
        load_reads_from_readtxt,
        load_gt_tags_file,
        load_gt_refs_fasta,
        build_tag_to_ref_mapping,
        match_reads_to_gt,
        find_read_txt,
    )
except ImportError as e:
    print("❌ 无法 import eval_reconstruction。自动回溯未找到 code/models/eval_reconstruction.py。")
    print("   请确认 spike 在 code 目录树下(如 CC/Step0/spikes/), 且 models/ 里有该文件。")
    print(f"   detail: {e}")
    sys.exit(1)


# ── 真实 MV 重建 (与 step2_decode.compute_mv_consensus / ds_fusion_masked 同口径) ──
def mv_consensus(read_seqs, ref_length):
    """
    逐位多数投票, 生成长度 = ref_length 的 consensus 字符串。
    与 compute_mv_consensus 同口径:
      - 每位置统计 ACGT 计数
      - has_vote: 有效(非越界) read 数 >= 50% * N 才保留该位置, 否则该位置丢弃
      - argmax 取多数碱基
    注: compute_mv_consensus 的 padding 是 one-hot 越界=全零, 这里等价用 "read 长度不足该位"。
    """
    N = len(read_seqs)
    if N == 0:
        return ""
    thresh = max(N * 0.5, 1)
    out = []
    for pos in range(ref_length):
        counter = Counter()
        valid = 0
        for s in read_seqs:
            if pos < len(s):
                b = s[pos]
                valid += 1
                if b in 'ACGT':
                    counter[b] += 1
        if valid >= thresh and counter:
            out.append(counter.most_common(1)[0][0])
        # else: 该位置无足够投票 -> 跳过(等价 padding 清零, consensus 变短)
    return ''.join(out)


def main():
    ap = argparse.ArgumentParser(
        description="欠分割 vs 过分割诊断 spike (只读, 真实MV, 同eval口径)")
    ap.add_argument('--experiment_dir', required=True)
    ap.add_argument('--gt_refs', required=True, help='reads.fasta (干净GT reference)')
    ap.add_argument('--gt_tags', required=True, help='output.txt (tag<TAB>read)')
    ap.add_argument('--read_txt', default=None, help='read.txt (默认自动发现)')
    ap.add_argument('--ref_length', type=int, default=196,
                    help='参考序列长度(Seq_1D=196), MV 截断长度')
    ap.add_argument('--min_reads_success', type=int, default=1,
                    help='单独MV判success时该GT在簇内最少read数(默认1, 即只要有read就尝试)')
    args = ap.parse_args()

    print("=" * 72)
    print("  欠分割 vs 过分割 诊断 spike  (只读 / 真实MV / 同 eval_reconstruction 口径)")
    print("=" * 72)
    print(f"  ref_length = {args.ref_length}")

    # ── 1. reads + Clover 初始标签 ────────────────────────────────────────────
    read_txt = args.read_txt or find_read_txt(args.experiment_dir)
    if read_txt is None:
        print("❌ 找不到 read.txt, 请用 --read_txt 指定"); sys.exit(1)
    print(f"\n[1] 加载 reads: {read_txt}")
    reads, clover_labels = load_reads_from_readtxt(read_txt)

    # ── 2. GT 链路 (与 eval 完全一致) ─────────────────────────────────────────
    print(f"\n[2] 加载 GT tags: {args.gt_tags}")
    seq_to_tag, tag_to_reads = load_gt_tags_file(args.gt_tags)
    print(f"\n[3] 加载 GT refs: {args.gt_refs}")
    gt_refs = load_gt_refs_fasta(args.gt_refs)
    ref_len_med = int(np.median([len(s) for s in gt_refs.values()]))
    print(f"    GT ref 中位长度: {ref_len_med}bp")
    tag_to_ref = build_tag_to_ref_mapping(tag_to_reads, gt_refs, ref_len=ref_len_med)
    print(f"\n[4] 匹配 reads -> GT ref id")
    gt_ref_ids = match_reads_to_gt(reads, seq_to_tag, tag_to_ref)  # per-read ref_id, -1=未匹配

    n_reads = len(reads)
    has_gt = gt_ref_ids >= 0
    print(f"    有GT的 read: {has_gt.sum():,}/{n_reads:,} ({has_gt.sum()/n_reads*100:.1f}%)")

    # ── 3. 建簇内结构 ─────────────────────────────────────────────────────────
    # cluster -> list of read global idx
    cl_to_ridx = defaultdict(list)
    for i, c in enumerate(clover_labels):
        cl_to_ridx[int(c)].append(i)

    # GT 全集(只数在 reads 里出现过的 GT, 与 eval 的 covered 视角一致)
    all_gt = set(int(g) for g in gt_ref_ids[has_gt].tolist())
    n_gt = len(all_gt)
    n_clusters = len(cl_to_ridx)
    print(f"\n[5] 结构概览")
    print(f"    reads 中出现的 GT 分子数 : {n_gt:,}")
    print(f"    Clover 簇数              : {n_clusters:,}")
    print(f"    净差 (簇数 - GT数)       : {n_clusters - n_gt:+,}  "
          f"({'簇多→偏过分割' if n_clusters>n_gt else '簇少→偏欠分割'})")

    # 每簇内 GT 分布 + 主GT
    cl_gt_counts = {}     # cluster -> Counter(gt_id -> #read)  (只数有GT的read)
    cl_majgt = {}         # cluster -> 主GT id
    for c, ridxs in cl_to_ridx.items():
        cnt = Counter()
        for ri in ridxs:
            g = int(gt_ref_ids[ri])
            if g >= 0:
                cnt[g] += 1
        cl_gt_counts[c] = cnt
        if cnt:
            cl_majgt[c] = cnt.most_common(1)[0][0]

    pure = sum(1 for c, cnt in cl_gt_counts.items() if len(cnt) == 1)
    mixed = sum(1 for c, cnt in cl_gt_counts.items() if len(cnt) >= 2)
    empty = sum(1 for c, cnt in cl_gt_counts.items() if len(cnt) == 0)
    print(f"    纯簇(1个GT)              : {pure:,}")
    print(f"    混合簇(>=2 GT)           : {mixed:,}")
    print(f"    空簇(无GT read)          : {empty:,}")

    # GT -> 它出现在哪些簇 (用于判过分割: 一个GT被切到多簇)
    gt_to_clusters = defaultdict(list)
    for c, cnt in cl_gt_counts.items():
        for g in cnt:
            gt_to_clusters[g].append(c)
    split_gts = [g for g, cs in gt_to_clusters.items() if len(cs) >= 2]
    print(f"    被切成多簇的 GT (过分割) : {len(split_gts):,}  "
          f"({len(split_gts)/n_gt*100:.1f}% of GT)")

    # ── 4. 现状 baseline: 现有簇下多少 GT 已 success ──────────────────────────
    # 与 eval 口径: 每簇 majority vote -> 主GT; 簇 consensus = 簇内全部read MV;
    # 若 consensus == 该主GT的真ref(ED==0) 则该主GT success(取该GT所有承载簇的最优)。
    print(f"\n[6] 现状 baseline (现有 Clover 聚类, 同 eval 口径自测)")
    print(f"    (注: 此数应与你 eval R0/Clover SR 近似, 用作交叉验证口径)")

    # 缓存每簇的 consensus (全簇 read MV)
    def cluster_seqs(c):
        return [reads[ri] for ri in cl_to_ridx[c]]

    gt_success_now = set()
    # 对每个 GT, 它的承载簇 = 以它为主GT的簇 (与 build_cluster_to_gt 一致)
    majgt_to_clusters = defaultdict(list)
    for c, mg in cl_majgt.items():
        majgt_to_clusters[mg].append(c)

    for g in all_gt:
        cs = majgt_to_clusters.get(g, [])
        if not cs:
            continue
        ok = False
        for c in cs:
            cons = mv_consensus(cluster_seqs(c), args.ref_length)
            if cons and levenshtein(cons, gt_refs[g]) == 0:
                ok = True
                break
        if ok:
            gt_success_now.add(g)

    sr_now = len(gt_success_now) / n_gt
    print(f"    现状 success GT: {len(gt_success_now):,}/{n_gt:,}  (自测SR={sr_now:.4f})")

    # ── 5. 过分割侧: 合并可救上界 ─────────────────────────────────────────────
    # 对每个被切GT: 它的所有承载簇(以它为主GT的簇)各自单独MV是否success;
    # 若全部fail, 但所有承载簇read合起来MV success -> 合并可救(+1)。
    print(f"\n[7] 过分割侧 — 合并可救上界")
    merge_rescuable = []
    for g in split_gts:
        cs = majgt_to_clusters.get(g, [])
        if len(cs) < 2:
            continue  # 必须有>=2个以它为主的碎片簇才谈"合并"
        if g in gt_success_now:
            continue  # 现状已success, 合并不新增
        # 各碎片单独
        any_frag_ok = False
        merged_reads = []
        for c in cs:
            seqs = cluster_seqs(c)
            merged_reads.extend(seqs)
            cons = mv_consensus(seqs, args.ref_length)
            if cons and levenshtein(cons, gt_refs[g]) == 0:
                any_frag_ok = True
                break
        if any_frag_ok:
            continue  # 已有单碎片能成, 不算"靠合并才救"
        merged_cons = mv_consensus(merged_reads, args.ref_length)
        if merged_cons and levenshtein(merged_cons, gt_refs[g]) == 0:
            merge_rescuable.append(g)
    print(f"    被切GT(主簇>=2)中, 碎片各自fail但合并后success: {len(merge_rescuable):,}")
    print(f"    -> 合并可救上界 ΔSR(success) = +{len(merge_rescuable)}  "
          f"(+{len(merge_rescuable)/n_gt*100:.3f} pt)")

    # ── 6. 欠分割侧: 拆分可救上界 ─────────────────────────────────────────────
    # 对每个混合簇, 主GT外的每个次要GT: 抽其在簇内read单独MV, 若success且该GT
    # 现状未success -> 拆分可救(+1)。一个GT可能在多个混合簇当次要, 只要任一处能拆出即算救。
    print(f"\n[8] 欠分割侧 — 拆分可救上界")
    split_rescuable = set()
    secondary_total = 0
    for c, cnt in cl_gt_counts.items():
        if len(cnt) < 2:
            continue
        mg = cl_majgt[c]
        # 簇内每个read的global idx, 按GT分组
        gt_to_local = defaultdict(list)
        for ri in cl_to_ridx[c]:
            g = int(gt_ref_ids[ri])
            if g >= 0:
                gt_to_local[g].append(ri)
        for g, ris in gt_to_local.items():
            if g == mg:
                continue  # 主GT不算被压制
            secondary_total += 1
            if g in gt_success_now or g in split_rescuable:
                continue
            if len(ris) < args.min_reads_success:
                continue
            seqs = [reads[ri] for ri in ris]
            cons = mv_consensus(seqs, args.ref_length)
            if cons and levenshtein(cons, gt_refs[g]) == 0:
                split_rescuable.add(g)
    print(f"    混合簇中 (簇,次要GT) 对数 : {secondary_total:,}")
    print(f"    其中次要GT单独MV能success且现状未success: {len(split_rescuable):,}")
    print(f"    -> 拆分可救上界 ΔSR(success) = +{len(split_rescuable)}  "
          f"(+{len(split_rescuable)/n_gt*100:.3f} pt)")

    # ── 7. 净对比 + 覆盖影响 ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  结论对比")
    print(f"{'='*72}")
    print(f"  现状 self-SR              : {sr_now:.4f}  ({len(gt_success_now):,}/{n_gt:,})")
    print(f"  合并可救上界 (过分割侧)   : +{len(merge_rescuable):,}  "
          f"-> SR 至多 {(len(gt_success_now)+len(merge_rescuable))/n_gt:.4f}")
    print(f"  拆分可救上界 (欠分割侧)   : +{len(split_rescuable):,}  "
          f"-> SR 至多 {(len(gt_success_now)+len(split_rescuable))/n_gt:.4f}")
    print()
    if len(split_rescuable) > len(merge_rescuable):
        print(f"  ➜ 拆分混合簇 天花板更高 ({len(split_rescuable)} > {len(merge_rescuable)})。"
              f"欠分割是主损失, 拆分是提SR的钥匙。")
    elif len(merge_rescuable) > len(split_rescuable):
        print(f"  ➜ 合并碎片簇 天花板更高 ({len(merge_rescuable)} > {len(split_rescuable)})。"
              f"过分割是主损失, edit合并方向应保留。")
    else:
        print(f"  ➜ 两侧相当, 拆/合都救不回多少, SR 已接近 Clover 结构上限。")
    total_ceiling = (len(gt_success_now)+len(merge_rescuable)+len(split_rescuable))/n_gt
    print(f"  两侧都做的理论上限 SR     : {total_ceiling:.4f}")
    print(f"  (对照: 现状 {sr_now:.4f} / FedDNA完美簇 0.9726)")

    # 覆盖影响(read_util视角): 拆分不改变 -1, 不影响 read_util; 合并也不改 -1。
    # 这两个方向都只重组已分配read, 故对 read_util 中性。塌陷另由 Zone III(已关)。
    print(f"\n  [覆盖说明] 拆分/合并均只重组已分配read, 不产生-1, 对 read_util 中性。")
    print(f"            => 提升只体现在 RQS(质量轴)/SR, 不在覆盖轴。")
    print("=" * 72)


if __name__ == "__main__":
    main()