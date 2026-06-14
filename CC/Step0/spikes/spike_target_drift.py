#!/usr/bin/env python
"""
spike_target_drift.py —— 验证"重建靶点逐轮变脏"假设
=====================================================
假设: SR 从 R1(0.9139) 退化到 R3(0.876) 的原因不是 embedding (已证保距 0.999),
      而是每轮的 consensus 训练靶点在迭代中越变越脏, 把重建头越教越歪.

方法: 对 R1/R2/R3 三轮的 consensus_dict, 把每个簇的 consensus 序列
      和它对应的真 GT reference 算归一化 edit 距离, 看是否逐轮变大.

对齐链 (关键, 三套编号要打通):
  consensus_dict 的 key = Clover 簇 id
  -> 用簇内 read 的多数真 GT (majority gt_label) 确定这个簇对应哪个 GT 分子
  -> 用该 GT 分子编号去 reads.fasta 取真 reference 序列
  -> consensus 序列 vs 真 reference 算 norm edit

判读:
  - R1<R2<R3 单调变大       -> 靶点逐轮变脏, 真根因坐实. 修复=固定靶点/冻结重建头
  - 三轮基本持平            -> 靶点没变脏, 退化在别处(Step2 标签赋值), 转查那条线
  - 数值整体就很大(>0.05)   -> 靶点从一开始就不准, 是另一类问题

依赖: pip 安装 edlib (你环境里 spike_g 用过 edlib, 应该有)
只读, 不碰训练.
"""
import os, sys, glob, argparse
from collections import defaultdict, Counter
import numpy as np
import torch

BASE = ['A', 'C', 'G', 'T']


def decode_consensus(one_hot):
    """Tensor(L,4) one-hot -> 序列字符串. padding(全零行)跳过."""
    oh = one_hot
    if not torch.is_tensor(oh):
        oh = torch.as_tensor(oh)
    mask = oh.sum(dim=-1) > 0          # 有效位置
    idx = oh.argmax(dim=-1)            # (L,)
    chars = [BASE[int(idx[i])] for i in range(len(idx)) if bool(mask[i])]
    return ''.join(chars)


def norm_edit(a, b):
    import edlib
    if len(a) == 0 or len(b) == 0:
        return 1.0
    d = edlib.align(a, b, task='distance')['editDistance']
    return d / max(len(a), len(b))


def read_fasta_ordered(path):
    """>N / seq -> list[seq], 按 N 升序. reads.fasta 的 >1..>K 对应
    BWA @SQ SN 升序的第 1..K 个 reference."""
    pairs = []
    cur = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                cur = int(line[1:].strip())
            else:
                if cur is not None:
                    pairs.append((cur, line.upper()))
                    cur = None
    pairs.sort(key=lambda x: x[0])      # 按 >N 升序
    return [seq for _, seq in pairs]    # list, index 0 = >1


def build_tag_to_ref(sam_path, ref_fasta_path):
    """对齐铁律实现:
       BWA @SQ 的 SN (=output.txt 的 tag) 升序排列, 第 k 个 SN
       对应 reads.fasta 的第 k 条 (>k). 返回 {tag(int): ref_seq}.
    """
    # 1. 读 @SQ 得到所有 SN, 升序
    sns = []
    with open(sam_path) as f:
        for line in f:
            if line.startswith('@SQ'):
                # @SQ\tSN:11\tLN:196
                for field in line.strip().split('\t'):
                    if field.startswith('SN:'):
                        sns.append(int(field[3:]))
                        break
            elif not line.startswith('@'):
                break  # header 结束
    sns.sort()
    # 2. 读 reads.fasta 按 >N 升序
    ref_seqs = read_fasta_ordered(ref_fasta_path)
    # 3. 第 k 个 SN <-> 第 k 条 ref
    n = min(len(sns), len(ref_seqs))
    tag_to_ref = {sns[k]: ref_seqs[k] for k in range(n)}
    print(f"   [对齐] @SQ SN 数={len(sns)}, reads.fasta 条数={len(ref_seqs)}, "
          f"映射建立={n} 条")
    return tag_to_ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment_dir', required=True)
    p.add_argument('--gt_tags', required=True)
    p.add_argument('--ref_fasta', required=True,
                   help='reads.fasta (>N/seq), >k 对应排序SN第k个')
    p.add_argument('--sam', required=True,
                   help='mem-se.sam, 读 @SQ 得到 SN(=tag) 升序列表')
    p.add_argument('--code_dir', default='/mnt/st_data/liangxinyi/code')
    p.add_argument('--n_clusters_eval', type=int, default=2000,
                   help='每轮随机抽多少个簇算 edit (全量太慢)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.code_dir not in sys.path:
        sys.path.insert(0, args.code_dir)
    from models.step1_data import CloverDataLoader

    rng = np.random.default_rng(args.seed)

    # ---- 1. 加载 reads + GT (序列做 key, 100% 匹配那套) ----
    print("=" * 60); print("📂 加载 reads + GT"); print("=" * 60)
    # 用最新 refined_labels 让 loader 正常初始化即可; GT 用 load_gt_tags
    latest_labels = sorted(glob.glob(os.path.join(
        args.experiment_dir, '04_Iterative_Labels', 'refined_labels_*.txt')),
        key=os.path.getmtime)
    dl = CloverDataLoader(args.experiment_dir,
                          labels_path=latest_labels[-1] if latest_labels else None)
    dl.load_gt_tags(args.gt_tags)
    gt = np.array(dl.gt_labels)
    N = len(dl.reads)
    print(f"   reads={N}, GT匹配={int((gt>=0).sum())}")

    # ---- 2. 建 tag -> 真 GT reference 映射 (排序SN排名对齐) ----
    ref = build_tag_to_ref(args.sam, args.ref_fasta)
    print(f"   真 reference: {len(ref)} 条 tag->ref 映射")
    # sanity check: GT tag 现在应该几乎都能在 tag->ref 映射里找到
    gt_vals = sorted(set(int(x) for x in gt[gt >= 0].tolist()))
    overlap = len(set(gt_vals) & set(ref.keys()))
    print(f"   GT tag 数={len(gt_vals)}, 能找到 ref 的={overlap} "
          f"({overlap/max(len(gt_vals),1)*100:.1f}%)  <-- 应接近100%")
    if overlap < len(gt_vals) * 0.9:
        print("   ❗ 交集仍低于90%, 对齐可能有残留问题, 谨慎看结果。")

    # ---- 3. 找三轮 consensus_dict (按时间排序 = R1,R2,R3) ----
    cons_paths = sorted(glob.glob(os.path.join(
        args.experiment_dir, '04_Iterative_Labels', 'consensus_dict_*.pt')),
        key=os.path.getmtime)
    lbl_paths = sorted(glob.glob(os.path.join(
        args.experiment_dir, '04_Iterative_Labels', 'refined_labels_*.txt')),
        key=os.path.getmtime)
    print(f"\n   找到 {len(cons_paths)} 个 consensus_dict, {len(lbl_paths)} 个 labels")
    for i, c in enumerate(cons_paths):
        print(f"     R{i+1}: {os.path.basename(c)}")

    n_round = min(len(cons_paths), len(lbl_paths))
    if n_round == 0:
        print("❌ 没找到 consensus_dict"); return

    print("\n" + "=" * 60)
    print("🔬 逐轮: consensus 靶点 vs 真 GT reference 的归一化 edit")
    print("=" * 60)

    results = []
    for r in range(n_round):
        cons = torch.load(cons_paths[r], map_location='cpu')
        cons = {int(k): v for k, v in cons.items()}
        labels_full = np.loadtxt(lbl_paths[r], dtype=int)  # 全量 read 的簇标签 (-1 噪声)

        # 簇 -> 多数 GT
        cluster_gt = defaultdict(Counter)
        for ridx in range(len(labels_full)):
            cl = labels_full[ridx]
            if cl >= 0 and ridx < len(gt) and gt[ridx] >= 0:
                cluster_gt[int(cl)][int(gt[ridx])] += 1

        # 抽样簇
        eval_cids = [c for c in cons.keys() if c in cluster_gt]
        rng.shuffle(eval_cids)
        eval_cids = eval_cids[:args.n_clusters_eval]

        eds = []
        n_no_ref = 0
        for cid in eval_cids:
            maj_gt = cluster_gt[cid].most_common(1)[0][0]
            if maj_gt not in ref:
                n_no_ref += 1
                continue
            cons_seq = decode_consensus(cons[cid])
            ref_seq = ref[maj_gt]
            eds.append(norm_edit(cons_seq, ref_seq))
        eds = np.array(eds)
        if len(eds) == 0:
            print(f"   R{r+1}: ❌ 无可比簇 (ref 对不上)")
            results.append(None)
            continue
        sr_proxy = (eds < 1e-9).mean()  # edit=0 的比例 = 完美重建率 ≈ SR 代理
        print(f"   R{r+1}: n={len(eds)}  "
              f"mean_edit={eds.mean():.4f}  median={np.median(eds):.4f}  "
              f"完美(edit=0)={sr_proxy*100:.1f}%  无ref跳过={n_no_ref}")
        results.append((eds.mean(), np.median(eds), sr_proxy))

    # ---- 判读 ----
    print("\n" + "=" * 60); print("📊 判读"); print("=" * 60)
    valid = [(i, x) for i, x in enumerate(results) if x is not None]
    if len(valid) >= 2:
        means = [x[0] for _, x in valid]
        srs = [x[2] for _, x in valid]
        print(f"   mean_edit 逐轮: {[f'{m:.4f}' for m in means]}")
        print(f"   完美率逐轮:    {[f'{s*100:.1f}%' for s in srs]}")
        if all(means[i] <= means[i+1] for i in range(len(means)-1)) and means[-1] > means[0] + 0.01:
            print("   🟢 靶点逐轮变脏 -> 真根因坐实在'重建靶点污染'")
            print("      修复方向: 固定靶点(用R1)/纯MV不掺模型/冻结重建头")
        elif abs(means[-1] - means[0]) < 0.01:
            print("   🟡 靶点基本没变 -> 退化不在靶点, 转查 Step2 标签赋值")
        else:
            print("   ⚪ 趋势不单调, 看上面逐轮数字具体分析")
    else:
        print("   ⚠️ 有效轮次不足, 检查 ref 对齐 (上面的交集比例)")


if __name__ == '__main__':
    main()