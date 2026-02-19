# models/post_process.py
"""
SSI-EC Post-processing: 全量距离分配

在所有 T 轮迭代结束后:
  1. 加载最终模型, 对所有 label==-1 的 reads 做推理得到 embedding
  2. 无条件分配到最近的簇质心 (不设 delta 阈值)
  3. 输出 final_labels.txt (全量, 无 -1)

理论依据:
  - DNA 存储中每条 read 必然来自某个 reference, 不存在"不属于任何簇"
  - 最后一轮质心来自迭代优化后的高质量 Zone I reads, 比 Clover 树索引中心更准确
  - 迭代过程中保持 -1 丢弃 (避免噪声污染训练), 只在最终评估时全量分配
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel
from models.step1_data import CloverDataLoader, seq_to_onehot


# ===========================================================================
# 噪声 Reads 专用 Dataset
# ===========================================================================
class NoiseReadDataset(Dataset):
    """只加载 label == -1 的 reads, 用于 post-processing 推理"""

    def __init__(self, data_loader, noise_indices, max_len=150):
        """
        Args:
            data_loader:   CloverDataLoader 实例
            noise_indices: list[int], label == -1 的全局 read 索引
            max_len:       序列最大长度
        """
        self.data_loader = data_loader
        self.noise_indices = noise_indices
        self.max_len = max_len

    def __len__(self):
        return len(self.noise_indices)

    def __getitem__(self, idx):
        real_idx = self.noise_indices[idx]
        seq = self.data_loader.reads[real_idx]
        encoding = seq_to_onehot(seq, self.max_len)
        return {
            'encoding': encoding,
            'read_idx': real_idx,
        }


# ===========================================================================
# 核心: 全量距离分配
# ===========================================================================
@torch.no_grad()
def post_process_final_assignment(experiment_dir, final_checkpoint_path,
                                  final_labels_path, centroids_path,
                                  output_dir, device='cuda',
                                  dim=256, max_length=150,
                                  gt_tags_file=None):
    """
    Post-processing: 对所有 label==-1 的 reads 做无条件最近邻分配

    Args:
        experiment_dir:       实验根目录
        final_checkpoint_path: 最终轮次的 model checkpoint
        final_labels_path:    最终轮次的 refined_labels.txt
        centroids_path:       最终轮次保存的 centroids.pt
        output_dir:           输出目录
        device:               计算设备
        dim:                  模型维度
        max_length:           序列最大长度
        gt_tags_file:         GT 标签文件 (可选)

    Returns:
        final_labels_path: str, 最终标签文件路径
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🔧 Post-processing: 全量距离分配")
    print(f"{'='*70}")

    # =====================================================================
    # 1. 加载标签, 识别 -1 reads
    # =====================================================================
    labels = np.loadtxt(final_labels_path, dtype=int)
    TOTAL_READS = len(labels)
    noise_mask = (labels == -1)
    n_noise = noise_mask.sum()

    print(f"   总 reads: {TOTAL_READS:,}")
    print(f"   label==-1 的 reads: {n_noise:,} ({n_noise/TOTAL_READS*100:.2f}%)")

    if n_noise == 0:
        print(f"   ✅ 无需 post-processing, 所有 reads 已有标签")
        final_path = os.path.join(output_dir, "final_labels.txt")
        np.savetxt(final_path, labels, fmt='%d')
        return final_path

    # =====================================================================
    # 2. 加载质心
    # =====================================================================
    centroids_data = torch.load(centroids_path, map_location='cpu')
    centroids = centroids_data['centroids']  # dict {cluster_id: tensor(D,)}
    print(f"   质心数: {len(centroids)}")

    # =====================================================================
    # 3. 加载模型
    # =====================================================================
    checkpoint = torch.load(final_checkpoint_path, map_location=device)
    step1_args = checkpoint.get('args', {})
    model_dim = step1_args.get('dim', dim)
    model_max_len = step1_args.get('max_length', max_length)

    num_clusters = max(50, len(centroids))
    model = Step1EvidentialModel(
        dim=model_dim, max_length=model_max_len,
        num_clusters=num_clusters, device=str(device)
    ).to(device)

    sd = checkpoint['model_state_dict']
    if 'length_adapter.weight' in sd:
        sh = sd['length_adapter.weight'].shape
        if sh[0] == model_max_len:
            model.length_adapter = nn.Linear(sh[1], sh[0]).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"   ✅ 模型加载完成")

    # =====================================================================
    # 4. 加载数据, 创建噪声 Dataset
    # =====================================================================
    data_loader = CloverDataLoader(experiment_dir, labels_path=final_labels_path)
    noise_indices = np.where(noise_mask)[0].tolist()

    noise_dataset = NoiseReadDataset(data_loader, noise_indices, model_max_len)
    noise_loader = DataLoader(
        noise_dataset, batch_size=1024, shuffle=False,
        num_workers=0, pin_memory=True
    )

    print(f"   📦 噪声 reads dataset: {len(noise_dataset)} samples")

    # =====================================================================
    # 5. 推理噪声 reads 的 embeddings
    # =====================================================================
    print(f"   🔮 推理噪声 reads...")
    t0 = time.time()

    N = len(noise_dataset)
    D = model_dim
    noise_embeddings = torch.zeros(N, D)
    noise_read_indices = torch.zeros(N, dtype=torch.long)
    offset = 0

    for batch in noise_loader:
        reads = batch['encoding'].to(device)
        idxs = batch['read_idx']
        B = reads.shape[0]

        # Padding / Truncation
        if reads.shape[1] != model_max_len:
            if reads.shape[1] < model_max_len:
                reads = F.pad(reads, (0, 0, 0, model_max_len - reads.shape[1]))
            else:
                reads = reads[:, :model_max_len, :]

        _, pooled = model.encode_reads(reads)
        noise_embeddings[offset:offset+B] = pooled.cpu()
        noise_read_indices[offset:offset+B] = idxs
        offset += B

    noise_embeddings = noise_embeddings[:offset]
    noise_read_indices = noise_read_indices[:offset].numpy()

    del model
    torch.cuda.empty_cache()

    t1 = time.time()
    print(f"   ✅ 推理完成: {offset} reads, {t1-t0:.1f}s")

    # =====================================================================
    # 6. 无条件最近邻分配
    # =====================================================================
    print(f"   📍 最近邻分配...")
    t0 = time.time()

    # Normalize embeddings
    noise_emb_norm = F.normalize(noise_embeddings, dim=-1)

    # 准备质心矩阵
    sorted_cids = sorted(centroids.keys())
    centroid_matrix = torch.stack([centroids[c] for c in sorted_cids])
    # 质心已在 compute_centroids_weighted 中 normalized

    # 分块计算最近邻 (避免 OOM)
    final_labels = labels.copy()
    chunk_size = 5000

    for i in range(0, len(noise_emb_norm), chunk_size):
        batch_emb = noise_emb_norm[i:i+chunk_size]
        batch_indices = noise_read_indices[i:i+chunk_size]

        # Cosine similarity → L2 distance
        sim = batch_emb @ centroid_matrix.T
        dist = torch.sqrt((2.0 - 2.0 * sim).clamp(min=0.0))
        nearest_idx = dist.argmin(dim=1)

        for j in range(len(batch_emb)):
            read_idx = int(batch_indices[j])
            cluster_id = sorted_cids[nearest_idx[j].item()]
            final_labels[read_idx] = cluster_id

    t1 = time.time()
    print(f"   ✅ 分配完成: {n_noise} reads → {len(sorted_cids)} 个簇, {t1-t0:.1f}s")

    # 验证: 不应该还有 -1
    remaining_noise = (final_labels == -1).sum()
    if remaining_noise > 0:
        print(f"   ⚠️ 仍有 {remaining_noise} 条 -1 (可能是初始就未被推理的 reads)")
    else:
        print(f"   ✅ 全量分配完成, 无残留 -1")

    # =====================================================================
    # 7. 保存
    # =====================================================================
    final_path = os.path.join(output_dir, "final_labels.txt")
    np.savetxt(final_path, final_labels, fmt='%d')
    print(f"   💾 最终标签: {final_path}")

    # =====================================================================
    # 8. 评估 (如果有 GT)
    # =====================================================================
    if gt_tags_file and os.path.exists(gt_tags_file):
        from models.eval_metrics import compute_all_metrics, save_metrics_report

        # 加载 GT
        if not hasattr(data_loader, 'gt_labels') or all(g == -1 for g in data_loader.gt_labels):
            data_loader.load_gt_tags(gt_tags_file)

        gt_labels_arr = np.array(data_loader.gt_labels)
        metrics = compute_all_metrics(final_labels, gt_labels_arr, verbose=True)

        report_path = os.path.join(output_dir, "final_eval_report.txt")
        save_metrics_report(metrics, report_path,
                            round_info="Post-processing final assignment")
    else:
        print(f"   ℹ️ 无 GT 文件, 跳过评估")

    return final_path