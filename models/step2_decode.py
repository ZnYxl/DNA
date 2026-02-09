# models/step2_decode.py
"""
Step2: Evidence-Weighted Consensus Decoding

修复清单:
  [FIX-#2]  情况2的 conf_weights 不再被 strength softmax 覆盖，
            改为两种权重相乘后归一化
"""
import torch
import torch.nn.functional as F


def decode_cluster_consensus(evidence, alpha, labels, strength, high_conf_mask,
                             verbose=False):
    """
    对每个有效簇，用高置信度 reads 的 α 做逐位投票恢复原始序列。

    Args:
        evidence:       (N, L, 4)
        alpha:          (N, L, 4)
        labels:         (N,) 修正后标签
        strength:       (N,) evidence strength
        high_conf_mask: (N,) bool 高置信度 mask
        verbose:        是否打印每个簇的详情 (大数据集建议 False)
    """
    consensus_dict = {}
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    unique_labels = torch.unique(labels)
    skipped_no_hc = 0

    for k in unique_labels:
        if k < 0:
            continue

        mask = (labels == k)
        count = mask.sum().item()
        if count < 2:
            continue

        cluster_alpha = alpha[mask]
        cluster_strength = strength[mask]
        cluster_high_conf = high_conf_mask[mask]
        high_conf_count = cluster_high_conf.sum().item()

        if high_conf_count == 0:
            skipped_no_hc += 1
            continue

        # ====== Consensus 策略 ======
        if high_conf_count >= 2:
            # 情况1: 足够的高置信度 reads → 简单平均
            consensus_alpha = cluster_alpha[cluster_high_conf]
            fused_alpha = consensus_alpha.mean(dim=0)
            if verbose:
                print(f"   🎯 簇{k}: {high_conf_count}/{count} 高置信度reads")
        else:
            # 情况2: 高置信度不足 → 加权融合
            # [FIX-#2] 两种权重相乘: conf 权重 × strength 权重
            conf_weights = torch.where(cluster_high_conf, 2.0, 0.5)
            str_weights  = F.softmax(cluster_strength, dim=0)
            combined     = conf_weights * str_weights
            combined     = (combined / combined.sum()).view(-1, 1, 1)
            fused_alpha  = torch.sum(cluster_alpha * combined, dim=0)
            if verbose:
                print(f"   ⚖️ 簇{k}: 加权融合 {high_conf_count}/{count}")

        # 解码为序列
        consensus_prob = fused_alpha / fused_alpha.sum(dim=-1, keepdim=True)
        consensus_indices = torch.argmax(consensus_prob, dim=-1)
        consensus_seq = ''.join([base_map[idx.item()] for idx in consensus_indices])

        consensus_dict[int(k.item())] = {
            'consensus_prob': consensus_prob.cpu(),
            'consensus_seq': consensus_seq,
            'num_reads': count,
            'num_high_conf': int(high_conf_count),
            'avg_strength': cluster_strength.mean().item()
        }

    print(f"\n   🧬 共识序列: {len(consensus_dict)} 个 (跳过 {skipped_no_hc} 个无高置信度簇)")
    if consensus_dict:
        avg_len = sum(len(v['consensus_seq']) for v in consensus_dict.values()) / len(consensus_dict)
        hc_ratios = [v['num_high_conf'] / v['num_reads'] for v in consensus_dict.values()]
        print(f"      平均长度: {avg_len:.1f}, 平均高置信度比: {sum(hc_ratios)/len(hc_ratios):.1%}")

    return consensus_dict


def save_consensus_sequences(consensus_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for label, info in sorted(consensus_dict.items()):
            f.write(f">cluster_{label}_reads{info['num_reads']}"
                    f"_highconf{info['num_high_conf']}"
                    f"_strength{info['avg_strength']:.3f}\n")
            f.write(f"{info['consensus_seq']}\n")
    print(f"   💾 共识序列已保存: {output_path}")


import os


def compute_consensus_quality_metrics(consensus_dict, gt_sequences=None):
    metrics = {
        'num_clusters': len(consensus_dict),
        'avg_reads_per_cluster': sum(c['num_reads'] for c in consensus_dict.values()) / max(len(consensus_dict), 1),
        'avg_strength': sum(c['avg_strength'] for c in consensus_dict.values()) / max(len(consensus_dict), 1),
        'avg_high_conf_ratio': sum(c['num_high_conf'] / c['num_reads'] for c in consensus_dict.values()) / max(len(consensus_dict), 1)
    }

    if gt_sequences is not None:
        matches = 0
        total = 0
        for label, info in consensus_dict.items():
            if label in gt_sequences:
                pred_seq = info['consensus_seq']
                gt_seq = gt_sequences[label]
                min_len = min(len(pred_seq), len(gt_seq))
                matches += sum(p == g for p, g in zip(pred_seq[:min_len], gt_seq[:min_len]))
                total += max(len(pred_seq), len(gt_seq))
        if total > 0:
            metrics['sequence_accuracy'] = matches / total

    return metrics