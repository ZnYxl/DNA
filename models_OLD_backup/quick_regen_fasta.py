#!/usr/bin/env python3
"""
quick_regen_fasta_v2.py
=======================
用"长度多数投票"重新生成 FASTA，不需要重跑训练。

根因: save_consensus_fasta 用 max(read_length) 截断。
      1 条 insertion read (197bp) 就拉长整个簇的 consensus。
      one_hot 存的是 F.one_hot 后的二值矩阵，尾部和真实位置无法区分。

修复: 加载 reads + labels，对每个 cluster 的 read 长度做多数投票 (mode)，
      以此作为 consensus 的正确长度。

用法:
  python quick_regen_fasta_v2.py \
      --read_txt    .../03_FedDNA_In/read.txt \
      --labels      .../refined_labels_162641.txt \
      --consensus   .../consensus_dict_162641.pt \
      --output      .../consensus_R3_fixed.fasta

  # 多轮:
  python quick_regen_fasta_v2.py \
      --read_txt    .../03_FedDNA_In/read.txt \
      --labels      .../refined_labels_140626.txt .../refined_labels_151851.txt .../refined_labels_162641.txt \
      --consensus   .../consensus_dict_140626.pt  .../consensus_dict_151851.pt  .../consensus_dict_162641.pt \
      --output      .../consensus_R1_fixed.fasta  .../consensus_R2_fixed.fasta  .../consensus_R3_fixed.fasta
"""
import torch
import numpy as np
import os
import argparse
from collections import Counter, defaultdict

BASE_MAP = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


def load_read_lengths(read_txt_path: str):
    """加载每条 read 的长度（不存储序列本身，省内存）。"""
    lengths = []
    with open(read_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("====="):
                continue
            lengths.append(len(line))
    print(f"   Reads: {len(lengths):,}")
    return lengths


def compute_cluster_consensus_lengths(labels: np.ndarray, read_lengths: list) -> dict:
    """
    对每个 cluster，用 read 长度的 mode（众数）作为 consensus 长度。

    为什么用 mode 而不是 median:
      GT reference 是固定长度 (e.g. 196bp)。
      大部分 reads 的长度在 [191, 201] 范围内（±5bp 过滤后）。
      mode 直接给出"最可能的 reference 长度"，不受离群值影响。
    """
    cluster_lengths = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            cluster_lengths[int(label)].append(read_lengths[i])

    consensus_len = {}
    for cid, lens in cluster_lengths.items():
        counter = Counter(lens)
        mode_len = counter.most_common(1)[0][0]
        consensus_len[cid] = mode_len

    return consensus_len


def save_fixed_fasta(consensus_dict, consensus_lengths, fasta_path):
    """用 length majority vote 截断 consensus，生成修复版 FASTA。"""
    os.makedirs(os.path.dirname(fasta_path) or '.', exist_ok=True)
    n_written = 0
    len_stats = []
    n_truncated = 0
    n_extended = 0

    with open(fasta_path, 'w') as ff:
        for cluster_id, one_hot in sorted(consensus_dict.items()):
            # 确定截断长度
            target_len = consensus_lengths.get(cluster_id)
            if target_len is None:
                # 没有 reads 信息，fallback: 用 one_hot 有效长度
                valid_mask = one_hot.sum(dim=-1) > 0
                target_len = int(valid_mask.sum().item())

            # 截断到 target_len
            oh = one_hot[:target_len]  # (target_len, 4)

            # 碱基解码
            indices = oh.argmax(dim=-1).numpy()
            seq = ''.join(BASE_MAP[i] for i in indices)
            ff.write(f">cluster_{cluster_id}\n{seq}\n")
            n_written += 1
            len_stats.append(len(seq))

            # 统计
            orig_len = int((one_hot.sum(dim=-1) > 0).sum().item())
            if target_len < orig_len:
                n_truncated += 1
            elif target_len > orig_len:
                n_extended += 1

    len_arr = np.array(len_stats)
    print(f"   写入 {n_written} 条 consensus")
    if len(len_arr) > 0:
        print(f"   长度分布: min={len_arr.min()}, "
              f"median={np.median(len_arr):.0f}, max={len_arr.max()}")
        for ref_len in [196]:
            exact = int((len_arr == ref_len).sum())
            longer = int((len_arr > ref_len).sum())
            shorter = int((len_arr < ref_len).sum())
            print(f"   {ref_len}bp: {exact}  >{ref_len}bp: {longer}  <{ref_len}bp: {shorter}")
    print(f"   截断 (尾部去噪): {n_truncated}  延长: {n_extended}  不变: {n_written - n_truncated - n_extended}")
    print(f"   💾 {fasta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="长度多数投票修复版 FASTA 生成 (不需要重跑训练)"
    )
    parser.add_argument('--read_txt', required=True,
                        help='read.txt 路径')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='refined_labels_XXXXXX.txt 路径 (一个或多个)')
    parser.add_argument('--consensus', nargs='+', required=True,
                        help='consensus_dict_XXXXXX.pt 路径 (一个或多个)')
    parser.add_argument('--output', nargs='+', required=True,
                        help='输出 FASTA 路径 (一个或多个)')
    args = parser.parse_args()

    if not (len(args.labels) == len(args.consensus) == len(args.output)):
        print("❌ --labels, --consensus, --output 数量必须一致")
        return

    # 加载 read 长度（只需加载一次）
    print(f"\n{'═' * 60}")
    print(f"  📂 加载 read 长度")
    print(f"{'═' * 60}")
    print(f"  路径: {args.read_txt}")
    read_lengths = load_read_lengths(args.read_txt)

    # 逐轮处理
    for labels_path, dict_path, out_path in zip(args.labels, args.consensus, args.output):
        print(f"\n{'═' * 60}")
        print(f"  📦 处理: {os.path.basename(dict_path)}")
        print(f"{'═' * 60}")

        if not os.path.exists(dict_path):
            print(f"  ⚠️ 文件不存在: {dict_path}")
            continue
        if not os.path.exists(labels_path):
            print(f"  ⚠️ 文件不存在: {labels_path}")
            continue

        # 加载 labels
        print(f"   Labels: {os.path.basename(labels_path)}")
        labels = np.loadtxt(labels_path, dtype=int)
        print(f"   Labels 长度: {len(labels):,}")

        if len(labels) != len(read_lengths):
            print(f"   ⚠️ labels 长度 {len(labels)} ≠ reads 长度 {len(read_lengths)}, 跳过")
            continue

        # 计算每个 cluster 的 consensus 长度
        consensus_lengths = compute_cluster_consensus_lengths(labels, read_lengths)
        len_values = list(consensus_lengths.values())
        print(f"   簇 consensus 长度: mode of modes = {Counter(len_values).most_common(1)[0]}")

        # 加载 consensus_dict
        consensus_dict = torch.load(dict_path, map_location='cpu')
        print(f"   Consensus dict: {len(consensus_dict)} 个簇")

        # 生成修复版 FASTA
        save_fixed_fasta(consensus_dict, consensus_lengths, out_path)

    print(f"\n{'═' * 60}")
    print("  ✅ 全部完成!")
    print(f"{'═' * 60}")
    print("\n  下一步: 用 eval_reconstruction_v2.py 验证修复效果")


if __name__ == '__main__':
    main()