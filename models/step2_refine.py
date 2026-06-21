# models/step2_refine.py
"""
Step2 helper: strength-weighted centroid computation.

簇质心计算: 每个簇内所有 read 按 evidential strength 加权平均。
strength 来自 Dirichlet evidence 的总证据量, 高质量 read 权重更高。

注: 历史版本中的三区划分 (split_confidence_by_zone)、全局 delta
(compute_global_delta)、Zone-aware 重分配 (refine_reads) 已移除 ——
在 "训练 + 簇内拆分" 的终态流程中, 这些机制对 label 与重建质量均无贡献。
"""
from collections import defaultdict


def compute_centroids_weighted(embeddings, labels, strength):
    """
    每个簇按 strength 加权平均 embedding, 得到簇质心。

    Args:
        embeddings: (N, D) float tensor
        labels:     (N,)   long tensor, -1 表示无标签 (跳过)
        strength:   (N,)   float tensor, 每条 read 的证据强度

    Returns:
        centroids:     {cluster_id: (D,) tensor}
        cluster_sizes: {cluster_id: int}
    """
    centroids = {}
    cluster_sizes = {}

    # O(N) 建倒排索引: label -> 该簇所有 read 下标
    label_to_idx = defaultdict(list)
    labels_cpu = labels.cpu().numpy()
    for i, l in enumerate(labels_cpu):
        if l >= 0:
            label_to_idx[int(l)].append(i)

    import torch
    for k_int, idxs in label_to_idx.items():
        mask_idx = torch.tensor(idxs, dtype=torch.long, device=embeddings.device)
        emb = embeddings[mask_idx]
        w   = strength[mask_idx]

        w_sum = w.sum()
        if w_sum < 1e-10:
            centroids[k_int] = emb.mean(dim=0)
        else:
            centroids[k_int] = (emb * w.unsqueeze(1)).sum(dim=0) / w_sum

        cluster_sizes[k_int] = len(idxs)

    print(f"   📍 质心 (strength 加权): {len(centroids)} 个簇")
    return centroids, cluster_sizes