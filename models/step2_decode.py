# models/step2_decode.py
"""
FedDNA-faithful Evidence Fusion Decoder

核心逻辑与 FedDNA models/Model.py 的 ds_fusion 完全一致:
    fused_evidence = mean(evidence_i, dim=reads)   ← 等权平均 evidence（非 alpha）
    alpha          = fused_evidence + 1
    prob           = alpha / sum(alpha)
    prediction     = argmax(prob, dim=-1)

与旧版的区别:
  - 旧版基于 high_conf_mask 做两阶段加权，偏离 FedDNA 原设计
  - 新版严格复用 ds_fusion，只做 padding mask（全零位置不参与均值）
  - 第二轮 pass 策略：step2_runner 推理完毕、labels 稳定后按簇调用本模块
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, Optional

BASE_MAP = ['A', 'C', 'G', 'T']


# ---------------------------------------------------------------------------
# 核心: ds_fusion  (与 FedDNA Model.py 保持完全一致)
# ---------------------------------------------------------------------------
def ds_fusion(evidence: torch.Tensor) -> torch.Tensor:
    """
    等权平均簇内所有 reads 的 evidence。

    Args:
        evidence: (N, L, 4)  N 条 reads 的 per-position softplus evidence
    Returns:
        fused:    (L, 4)     融合后的 evidence
    """
    return evidence.mean(dim=0)


def ds_fusion_masked(evidence: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """
    带 padding mask 的 ds_fusion。
    padding_mask: (N, L) bool，True = 该 read 该位置是有效碱基（非padding）

    对每个位置只对有效 reads 求均值；若某位置所有 reads 都是 padding，
    fused_evidence 该位置置 0（后续 has_vote 检测排除）。
    """
    # (N, L, 4) × (N, L, 1) → mask 无效位置
    mask_f = padding_mask.float().unsqueeze(-1)          # (N, L, 1)
    evidence_masked = evidence * mask_f                  # 无效位置 evidence 归零
    count = mask_f.sum(dim=0).clamp(min=1)               # (L, 1) 每位置有效 reads 数
    fused = evidence_masked.sum(dim=0) / count           # (L, 4)
    has_vote = (mask_f.sum(dim=0).squeeze(-1) > 0)       # (L,) 该位置是否有有效碱基
    return fused, has_vote


# ---------------------------------------------------------------------------
# 主函数: 按簇做第二轮推理 + evidence fusion → consensus_dict
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_feddna_decode(
    model,
    data_loader,
    new_labels_np: np.ndarray,          # step2_runner 输出的最终标签
    flat_real_indices,                   # dataset.valid_indices
    model_max_len: int,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[int, torch.Tensor]:
    """
    FedDNA 式 Evidence Fusion，替代 majority-vote consensus。

    流程:
        1. 按 new_labels 分组，建立 cluster_id → [real_idx, ...] 映射
        2. 对每个簇，批量推理所有 reads → evidence (N, L, 4)
        3. ds_fusion_masked → fused_evidence (L, 4)
        4. alpha = fused + 1 → prob = alpha/S → one_hot(argmax) 作为 consensus

    Returns:
        consensus_dict: {cluster_id: Tensor(L, 4)} one-hot，padding 位置全零
    """
    from models.step1_data import seq_to_onehot   # 复用现有 one-hot 编码

    model.eval()
    model.to(device)

    # ------------------------------------------------------------------
    # 1. 建立 cluster → real_idx 列表
    # ------------------------------------------------------------------
    cluster_to_ridx: Dict[int, list] = defaultdict(list)
    for didx, label in enumerate(new_labels_np):
        if label >= 0:
            real_idx = flat_real_indices[didx]
            cluster_to_ridx[int(label)].append(real_idx)

    print(f"\n🧬 FedDNA Evidence Fusion: {len(cluster_to_ridx)} 个簇")

    consensus_dict: Dict[int, torch.Tensor] = {}
    skipped = 0

    # ------------------------------------------------------------------
    # 2. 逐簇推理
    # ------------------------------------------------------------------
    for cluster_id, ridx_list in cluster_to_ridx.items():
        n_reads = len(ridx_list)
        if n_reads < 1:
            skipped += 1
            continue

        # 收集该簇所有 reads 的 one-hot encoding
        encodings = []
        padding_masks = []
        for ridx in ridx_list:
            seq = data_loader.reads[ridx]
            enc = seq_to_onehot(seq, model_max_len)       # (L, 4)
            encodings.append(enc)
            # padding mask: 有效碱基位置的 one-hot 行不全零
            pmask = enc.sum(dim=-1) > 0                   # (L,) bool
            padding_masks.append(pmask)

        enc_tensor  = torch.stack(encodings)              # (N, L, 4)
        pmask_tensor = torch.stack(padding_masks)         # (N, L)

        # 批量推理（防 OOM，分 batch）
        all_evidence = []
        for start in range(0, n_reads, batch_size):
            batch_enc = enc_tensor[start:start+batch_size].to(device)
            emb, _ = model.encode_reads(batch_enc)
            ev, _, _ = model.decode_to_evidence(emb)     # (bs, L, 4)
            all_evidence.append(ev.cpu())
            del batch_enc, emb, ev

        all_evidence = torch.cat(all_evidence, dim=0)    # (N, L, 4)

        # ------------------------------------------------------------------
        # 3. ds_fusion (与 FedDNA 完全一致) + padding mask
        # ------------------------------------------------------------------
        fused_evidence, has_vote = ds_fusion_masked(all_evidence, pmask_tensor)
        # fused_evidence: (L, 4)

        # ------------------------------------------------------------------
        # 4. 解码: alpha → prob → argmax → one_hot
        # ------------------------------------------------------------------
        alpha = fused_evidence + 1.0                      # Dirichlet 参数
        strength = alpha.sum(dim=-1, keepdim=True)        # (L, 1)
        prob = alpha / strength                           # (L, 4)

        indices = prob.argmax(dim=-1)                     # (L,)
        one_hot = F.one_hot(indices, num_classes=4).float()  # (L, 4)
        one_hot[~has_vote] = 0.0                         # padding 位置清零

        consensus_dict[cluster_id] = one_hot

    print(f"   ✅ 生成 {len(consensus_dict)} 个簇的 consensus (跳过 {skipped} 个空簇)")
    return consensus_dict


# ---------------------------------------------------------------------------
# 工具: 将 consensus_dict 保存为 FASTA（与 step2_runner 截断逻辑一致）
# ---------------------------------------------------------------------------
def save_consensus_fasta(
    consensus_dict: Dict[int, torch.Tensor],
    new_labels_np: np.ndarray,
    flat_real_indices,
    data_loader,
    model_max_len: int,
    fasta_path: str,
):
    """
    按簇内最大 read 长度截断，去掉尾部 padding（与 step2_runner FIX-FASTA 一致）。
    """
    # 计算每簇实际最大 read 长度
    cluster_actual_len: Dict[int, int] = {}
    for didx, label in enumerate(new_labels_np):
        if label >= 0:
            real_idx = flat_real_indices[didx]
            rl = min(len(data_loader.reads[real_idx]), model_max_len)
            cid = int(label)
            if cid not in cluster_actual_len or rl > cluster_actual_len[cid]:
                cluster_actual_len[cid] = rl

    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, 'w') as ff:
        for cluster_id, one_hot in sorted(consensus_dict.items()):
            actual_len = cluster_actual_len.get(cluster_id, model_max_len)
            indices = one_hot[:actual_len].argmax(dim=-1).numpy()
            seq = ''.join(BASE_MAP[i] for i in indices)
            ff.write(f">cluster_{cluster_id}\n{seq}\n")

    print(f"   💾 FedDNA Consensus FASTA: {fasta_path}")