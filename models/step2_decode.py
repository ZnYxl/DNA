# models/step2_decode.py
"""
Step2: Evidence-Weighted Consensus Decoding
核心：生成每个簇的最终共识序列
✅ 偏向确定性：高置信度reads主导consensus
"""
import torch
import torch.nn.functional as F


def decode_cluster_consensus(evidence, alpha, labels, strength, high_conf_mask):
    """
    ✅ Phase C: 偏向确定性的consensus解码
    高置信度reads 100%权重，低置信度reads仅作参考

    Args:
        evidence: (N, L, 4) 每条read的evidence
        alpha: (N, L, 4) Dirichlet参数
        labels: (N,) 修正后的标签
        strength: (N,) evidence strength
        high_conf_mask: (N,) 高置信度mask

    Returns:
        consensus_dict: dict[label] -> {
            'consensus_prob': (L, 4),
            'consensus_seq': str,
            'num_reads': int,
            'num_high_conf': int,
            'avg_strength': float
        }
    """
    consensus_dict = {}
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    unique_labels = torch.unique(labels)

    for k in unique_labels:
        if k < 0:  # 跳过噪声
            continue

        mask = (labels == k)
        count = mask.sum().item()

        if count < 2:
            continue

        # 该簇的reads
        cluster_alpha = alpha[mask]  # (cluster_size, L, 4)
        cluster_strength = strength[mask]  # (cluster_size,)
        cluster_high_conf = high_conf_mask[mask]  # (cluster_size,)

        high_conf_count = cluster_high_conf.sum().item()

        if high_conf_count == 0:
            print(f"   ⚠️ 簇{k}: 没有高置信度reads，跳过consensus")
            continue

        # ✅ 策略：优先使用高置信度reads
        if high_conf_count >= 2:
            # 只用高置信度reads做consensus
            consensus_alpha = cluster_alpha[cluster_high_conf]
            consensus_strength = cluster_strength[cluster_high_conf]
            print(f"   🎯 簇{k}: 只用 {high_conf_count}/{count} 高置信度reads")
        else:
            # 高置信度不够，加权融合（高置信度权重更大）
            weights = torch.where(cluster_high_conf, 2.0, 0.5)  # 高置信度2倍权重
            weights = weights / weights.sum()

            consensus_alpha = cluster_alpha
            consensus_strength = cluster_strength
            print(f"   ⚖️ 簇{k}: 加权融合 {high_conf_count}/{count} 高置信度reads")

        # Evidence-weighted fusion
        if high_conf_count >= 2:
            # 简单平均（高置信度reads质量相近）
            fused_alpha = consensus_alpha.mean(dim=0)  # (L, 4)
        else:
            # 加权融合
            weights = F.softmax(consensus_strength, dim=0).view(-1, 1, 1)
            fused_alpha = torch.sum(consensus_alpha * weights, dim=0)  # (L, 4)

        # 归一化得到概率分布
        consensus_prob = fused_alpha / fused_alpha.sum(dim=-1, keepdim=True)

        # 解码为序列
        consensus_indices = torch.argmax(consensus_prob, dim=-1)
        consensus_seq = ''.join([base_map[idx.item()] for idx in consensus_indices])

        consensus_dict[int(k.item())] = {
            'consensus_prob': consensus_prob.cpu(),
            'consensus_seq': consensus_seq,
            'num_reads': count,
            'num_high_conf': high_conf_count,
            'avg_strength': cluster_strength.mean().item()
        }

    print(f"\n   🧬 生成 {len(consensus_dict)} 个共识序列")
    if consensus_dict:
        avg_len = sum(len(info['consensus_seq']) for info in consensus_dict.values()) / len(consensus_dict)
        print(f"      平均长度: {avg_len:.1f}")

        high_conf_ratio = sum(info['num_high_conf'] / info['num_reads'] for info in consensus_dict.values()) / len(
            consensus_dict)
        print(f"      平均高置信度比例: {high_conf_ratio:.1%}")

    return consensus_dict


def save_consensus_sequences(consensus_dict, output_path):
    """
    保存共识序列为FASTA格式

    Args:
        consensus_dict: decode_cluster_consensus的输出
        output_path: str, 输出文件路径
    """
    with open(output_path, 'w') as f:
        for label, info in sorted(consensus_dict.items()):
            f.write(
                f">cluster_{label}_reads{info['num_reads']}_highconf{info['num_high_conf']}_strength{info['avg_strength']:.3f}\n")
            f.write(f"{info['consensus_seq']}\n")

    print(f"   💾 共识序列已保存: {output_path}")


def compute_consensus_quality_metrics(consensus_dict, gt_sequences=None):
    """
    计算共识序列质量指标

    Args:
        consensus_dict: decode_cluster_consensus的输出
        gt_sequences: dict[label] -> str, ground truth序列（可选）

    Returns:
        metrics: dict, 质量指标
    """
    metrics = {
        'num_clusters': len(consensus_dict),
        'avg_reads_per_cluster': sum(c['num_reads'] for c in consensus_dict.values()) / len(
            consensus_dict) if consensus_dict else 0,
        'avg_strength': sum(c['avg_strength'] for c in consensus_dict.values()) / len(
            consensus_dict) if consensus_dict else 0,
        'avg_high_conf_ratio': sum(c['num_high_conf'] / c['num_reads'] for c in consensus_dict.values()) / len(
            consensus_dict) if consensus_dict else 0
    }

    # 如果有GT，计算准确率
    if gt_sequences is not None:
        matches = 0
        total = 0
        for label, info in consensus_dict.items():
            if label in gt_sequences:
                pred_seq = info['consensus_seq']
                gt_seq = gt_sequences[label]
                matches += sum(p == g for p, g in zip(pred_seq, gt_seq))
                total += len(pred_seq)

        if total > 0:
            metrics['sequence_accuracy'] = matches / total

    return metrics