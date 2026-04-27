# models/step2_decode.py
"""
FedDNA-faithful Evidence Fusion Decoder

修复清单:
  [FIX-OOM] 逐簇循环末尾释放大 tensor，每 1000 簇清理一次 GPU 缓存

核心逻辑与 FedDNA models/Model.py 的 ds_fusion 完全一致（不改动）:
    fused_evidence = mean(evidence_i, dim=reads)   ← 等权平均 evidence（非 alpha）
    alpha          = fused_evidence + 1
    prob           = alpha / sum(alpha)
    prediction     = argmax(prob, dim=-1)

设计说明:
  FedDNA 使用等权平均是有意为之。Dirichlet evidence 本身已编码不确定性：
  高噪声 read 的 softplus evidence 值本来就低，等权平均时贡献自然小，
  无需额外的 strength 加权（否则是 double counting）。
  FedDNA 在 4 个数据集上平均 Success Rate 97.91%，保持原设计。
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, Optional

BASE_MAP = ['A', 'C', 'G', 'T']


# ---------------------------------------------------------------------------
# 核心: ds_fusion  (与 FedDNA Model.py 保持完全一致，不改动)
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


def ds_fusion_masked(evidence: torch.Tensor, padding_mask: torch.Tensor) -> tuple:
    """
    带 padding mask 的 ds_fusion（与 FedDNA 等权逻辑一致）。
    padding_mask: (N, L) bool，True = 该 read 该位置是有效碱基（非 padding）

    对每个位置只对有效 reads 求均值；若某位置所有 reads 都是 padding，
    fused_evidence 该位置置 0（后续 has_vote 检测排除）。
    """
    mask_f = padding_mask.float().unsqueeze(-1)          # (N, L, 1)
    evidence_masked = evidence * mask_f                  # 无效位置 evidence 归零
    count = mask_f.sum(dim=0).clamp(min=1)               # (L, 1) 每位置有效 reads 数
    fused = evidence_masked.sum(dim=0) / count           # (L, 4)
    # [Bug1-Fix] 多数投票门限: >= 50% reads 有碱基才保留，防止 1 条 insertion read 拉长序列
    N = evidence.shape[0]
    has_vote = (mask_f.sum(dim=0).squeeze(-1) >= max(N * 0.5, 1))  # (L,) bool
    return fused, has_vote


# ---------------------------------------------------------------------------
# 主函数: 按簇做推理 + evidence fusion → consensus_dict
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_feddna_decode(
    model,
    data_loader,
    new_labels_np: np.ndarray,
    flat_real_indices,
    model_max_len: int,
    device: torch.device,
    batch_size: int = 512,
    ref_length: int = None,      # [v5] 先验参考序列长度，截断 consensus
) -> Dict[int, torch.Tensor]:
    """
    FedDNA 式 Evidence Fusion。

    流程:
        1. 按 new_labels 分组，建立 cluster_id → [real_idx, ...] 映射
        2. 对每个簇，批量推理所有 reads → evidence (N, L, 4)
        3. ds_fusion_masked（等权，与 FedDNA 完全一致）
        4. alpha = fused + 1 → prob = alpha/S → one_hot(argmax) 作为 consensus

    注意：step2_runner 的全量推理阶段已经跑过一次 encoder，但只保留了
    pooled emb (N, D)，没有保留序列级 emb (N, L, D)（需要 ~210GB，存不下）。
    因此这里必须重跑一遍完整的 encoder → decoder，是架构上的必要代价，
    若后续有更大内存预算，可考虑缓存 float16 序列级 emb 来消除这次重复推理。

    Returns:
        consensus_dict: {cluster_id: Tensor(L, 4)} one-hot，padding 位置全零
    """
    from models.step1_data import seq_to_onehot

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

    print(f"\n🧬 FedDNA Evidence Fusion (等权): {len(cluster_to_ridx)} 个簇")

    consensus_dict: Dict[int, torch.Tensor] = {}
    skipped = 0

    # ------------------------------------------------------------------
    # 2. 逐簇推理
    # ------------------------------------------------------------------
    for proc_count, (cluster_id, ridx_list) in enumerate(cluster_to_ridx.items()):
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
            pmask = enc.sum(dim=-1) > 0                   # (L,) bool
            padding_masks.append(pmask)

        enc_tensor   = torch.stack(encodings)             # (N, L, 4)
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
        # 3. ds_fusion_masked（等权，与 FedDNA 完全一致）
        # ------------------------------------------------------------------
        fused_evidence, has_vote = ds_fusion_masked(all_evidence, pmask_tensor)

        # ------------------------------------------------------------------
        # 4. 解码: alpha → prob → argmax → one_hot
        # ------------------------------------------------------------------
        alpha    = fused_evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)        # (L, 1)
        prob     = alpha / strength                       # (L, 4)

        indices  = prob.argmax(dim=-1)                    # (L,)
        one_hot  = F.one_hot(indices, num_classes=4).float()
        one_hot[~has_vote] = 0.0                          # padding 位置清零

        consensus_dict[cluster_id] = one_hot

        # [FIX-OOM] 释放本簇的大 tensor，防止内存随簇数线性累积
        del enc_tensor, pmask_tensor, all_evidence, fused_evidence, one_hot

        # [FIX-OOM] 每处理 1000 个簇清理一次 GPU 缓存碎片
        if proc_count % 1000 == 999:
            torch.cuda.empty_cache()

    print(f"   ✅ 生成 {len(consensus_dict)} 个簇的 consensus (跳过 {skipped} 个空簇)")
    return consensus_dict



# ---------------------------------------------------------------------------
# [v16 路径B] MV Consensus: pure majority vote, 不跑 encoder
# ---------------------------------------------------------------------------
def compute_mv_consensus(
    data_loader,
    new_labels_np,
    flat_real_indices,
    model_max_len: int,
    ref_length: int = None,
) -> Dict[int, torch.Tensor]:
    """
    Pure majority-vote consensus, 用于 Round 2+ 的 Step1 训练靶子.

    与 run_feddna_decode 的区别:
      - 不跑 encoder/decoder, 纯统计投票
      - 不受 encoder 状态影响 -> 打破 encoder 自污染闭环
      - 输出格式完全一致 (Dict[int, Tensor(L, 4)] one-hot), step1_train.py 零改动

    逻辑:
      对每个簇 (只处理 label >= 0 的 reads):
        counts[L, 4] = sum over reads of one_hot(seq)
        has_vote[L]  = (有效 read 数 >= 50% * N)      # 与 ds_fusion_masked 一致
        indices[L]   = counts.argmax(dim=-1)
        one_hot[L,4] = F.one_hot(indices, 4); one_hot[~has_vote] = 0

    Args:
        data_loader:       CloverDataLoader, 提供 data_loader.reads[real_idx]
        new_labels_np:     严格版 labels (numpy, -1 会被自动跳过)
        flat_real_indices: data_loader.reads 的真实索引映射
        model_max_len:     序列 one-hot 长度 (与 fusion 版对齐, 通常 201)
        ref_length:        先验长度, 仅用于日志提示, 实际形状仍为 model_max_len
                           (与 run_feddna_decode 的 consensus_dict 对齐)

    Returns:
        consensus_dict: {cluster_id: Tensor(model_max_len, 4)} one-hot
    """
    from models.step1_data import seq_to_onehot

    # 1. cluster_id -> real_idx 列表 (只收集 label >= 0)
    cluster_to_ridx: Dict[int, list] = defaultdict(list)
    for didx, label in enumerate(new_labels_np):
        if label >= 0:
            real_idx = flat_real_indices[didx]
            cluster_to_ridx[int(label)].append(real_idx)

    print(f"\n🗳️  [v16 路径B] MV Consensus: {len(cluster_to_ridx)} 个簇")
    if ref_length is not None:
        print(f"   📏 ref_length={ref_length} (one-hot 形状仍为 L={model_max_len}, "
              f"截断在 save_consensus_fasta 处理)")

    consensus_dict: Dict[int, torch.Tensor] = {}
    skipped = 0

    # 2. 逐簇统计投票
    for cluster_id, ridx_list in cluster_to_ridx.items():
        n_reads = len(ridx_list)
        if n_reads < 1:
            skipped += 1
            continue

        # 收集 one-hot encoding + padding mask
        encodings = []
        padding_masks = []
        for ridx in ridx_list:
            seq = data_loader.reads[ridx]
            enc = seq_to_onehot(seq, model_max_len)   # (L, 4)
            encodings.append(enc)
            pmask = enc.sum(dim=-1) > 0               # (L,) bool
            padding_masks.append(pmask)

        enc_tensor   = torch.stack(encodings)          # (N, L, 4)
        pmask_tensor = torch.stack(padding_masks)      # (N, L)

        # 3. 统计投票
        counts   = enc_tensor.sum(dim=0)               # (L, 4)  每位置4碱基计数
        n_valid  = pmask_tensor.sum(dim=0).float()     # (L,)    每位置有效 read 数
        # has_vote: 投票门限与 ds_fusion_masked 一致 (防 1 条 insertion read 拉长)
        has_vote = (n_valid >= max(n_reads * 0.5, 1))  # (L,) bool

        # 4. argmax -> one_hot, padding 位置清零
        indices = counts.argmax(dim=-1)                # (L,)
        one_hot = F.one_hot(indices, num_classes=4).float()  # (L, 4)
        one_hot[~has_vote] = 0.0

        consensus_dict[cluster_id] = one_hot

        # 释放本簇的中间 tensor
        del enc_tensor, pmask_tensor, counts, n_valid, has_vote, indices, one_hot

    print(f"   ✅ 生成 {len(consensus_dict)} 个簇的 MV consensus "
          f"(跳过 {skipped} 个空簇)")
    return consensus_dict


# ---------------------------------------------------------------------------
# 工具: 将 consensus_dict 保存为 FASTA
# ---------------------------------------------------------------------------
def save_consensus_fasta(
    consensus_dict: Dict[int, torch.Tensor],
    new_labels_np: np.ndarray,
    flat_real_indices,
    data_loader,
    model_max_len: int,
    fasta_path: str,
    ref_length: int = None,
):
    """
    [Bug1-Fix-B] 用簇内 read 长度众数 (mode) 截断，替代 max(read_len)。

    原 Bug: max(read_len) 让 1 条 insertion read (197bp) 拉长整个簇，
    尾部全零 → argmax=0 → 多出一个 'A' → ED=1。
    修复: 长度众数 = "最可能的 reference 长度"，不受 insertion 离群值影响。
    """
    from collections import Counter as _Counter

    # 计算每个簇的 read 长度众数
    cluster_len_votes: Dict[int, list] = {}
    for didx, label in enumerate(new_labels_np):
        if label >= 0:
            real_idx = flat_real_indices[didx]
            rl = min(len(data_loader.reads[real_idx]), model_max_len)
            cid = int(label)
            if cid not in cluster_len_votes:
                cluster_len_votes[cid] = []
            cluster_len_votes[cid].append(rl)

    cluster_actual_len: Dict[int, int] = {}
    for cid, lens in cluster_len_votes.items():
        cluster_actual_len[cid] = _Counter(lens).most_common(1)[0][0]  # mode

    # [ref_length-FIX] 有先验时统一用先验长度，没有时退回众数
    if ref_length is not None:
        print(f"   📏 使用先验 ref_length={ref_length} 截断 (覆盖 read 众数)")
        mode_mismatch = sum(1 for cid, ml in cluster_actual_len.items() if ml != ref_length)
        if mode_mismatch > 0:
            print(f"   ⚠️ {mode_mismatch} 个簇的 read 众数 ≠ ref_length (将被先验覆盖)")

    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)
    with open(fasta_path, 'w') as ff:
        for cluster_id, one_hot in sorted(consensus_dict.items()):
            if ref_length is not None:
                actual_len = ref_length
            else:
                actual_len = cluster_actual_len.get(cluster_id, model_max_len)
            indices = one_hot[:actual_len].argmax(dim=-1).numpy()
            seq = ''.join(BASE_MAP[i] for i in indices)
            ff.write(f">cluster_{cluster_id}\n{seq}\n")

    print(f"   💾 FedDNA Consensus FASTA: {fasta_path}")