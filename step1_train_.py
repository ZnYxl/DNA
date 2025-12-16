#!/usr/bin/env python3
"""
FedDNA 簇序列重建系统 - 适配版 v3
=====================================
适配你的数据格式：
- read.txt: 按簇分组，用=======分隔
- ground_truth_clusters.txt: Cluster_ID \t Ref_Seq
- ground_truth_reads.txt: Read_ID \t Cluster_ID \t Ref_Seq \t Quality
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
@dataclass
class Config:
    """训练配置"""
    # 数据路径
    experiment_dir: str = ""  # 实验根目录
    
    # 模型参数
    input_dim: int = 6
    hidden_dim: int = 128
    latent_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    
    # 训练参数
    k_target: int = 50
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 1e-3
    
    # 损失权重
    lambda_contrastive: float = 1.0
    lambda_del: float = 0.5
    lambda_k: float = 2.0
    
    # 阈值
    similarity_threshold: float = 0.7
    min_cluster_ratio: float = 0.005
    weak_consistency_threshold: float = 0.6
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 数据加载 - 适配你的格式
# ============================================================================
def load_feddna_format(feddna_dir: str) -> Tuple[List[str], List[int]]:
    """
    加载 FedDNA 格式的数据
    read.txt: 按簇分组，用 =============================== 分隔
    
    返回: (reads列表, 簇标签列表)
    """
    read_path = os.path.join(feddna_dir, "read.txt")
    
    if not os.path.exists(read_path):
        raise FileNotFoundError(f"找不到 read.txt: {read_path}")
    
    reads = []
    labels = []
    current_cluster = 0
    
    print(f"📂 加载 FedDNA 格式数据: {read_path}")
    
    with open(read_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 检测分隔符
            if line.startswith("====="):
                current_cluster += 1
            else:
                # 这是一个read序列
                reads.append(line)
                labels.append(current_cluster)
    
    print(f"   ✅ 加载 {len(reads)} 条 reads")
    print(f"   ✅ 检测到 {current_cluster + 1} 个簇 (Clover聚类结果)")
    
    return reads, labels


def load_raw_reads_with_ids(raw_dir: str) -> Dict[str, str]:
    """
    加载原始reads (带ID)
    raw_reads.txt: Read_ID \t Sequence
    
    返回: {read_id: sequence}
    """
    raw_path = os.path.join(raw_dir, "raw_reads.txt")
    
    if not os.path.exists(raw_path):
        print(f"   ⚠️ raw_reads.txt 不存在")
        return {}
    
    reads_dict = {}
    with open(raw_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                reads_dict[parts[0]] = parts[1]
    
    print(f"   ✅ 加载 {len(reads_dict)} 条原始reads (带ID)")
    return reads_dict


def load_read_level_gt(raw_dir: str) -> Dict[str, Tuple[int, str, str]]:
    """
    加载 Read 级别的 GT
    ground_truth_reads.txt: Read_ID \t Cluster_ID \t Ref_Seq \t Quality
    
    返回: {read_id: (cluster_id, ref_seq, quality)}
    """
    gt_path = os.path.join(raw_dir, "ground_truth_reads.txt")
    
    if not os.path.exists(gt_path):
        print(f"   ⚠️ ground_truth_reads.txt 不存在")
        return {}
    
    gt_dict = {}
    with open(gt_path, 'r') as f:
        header = f.readline()  # 跳过表头
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                read_id = parts[0]
                cluster_id = int(parts[1])
                ref_seq = parts[2]
                quality = parts[3]
                gt_dict[read_id] = (cluster_id, ref_seq, quality)
    
    print(f"   ✅ 加载 {len(gt_dict)} 条 Read-Level GT")
    return gt_dict


def load_cluster_level_gt(raw_dir: str) -> Dict[int, str]:
    """
    加载 Cluster 级别的 GT
    ground_truth_clusters.txt: Cluster_ID \t Ref_Seq
    
    返回: {cluster_id: ref_seq}
    """
    gt_path = os.path.join(raw_dir, "ground_truth_clusters.txt")
    
    if not os.path.exists(gt_path):
        print(f"   ⚠️ ground_truth_clusters.txt 不存在")
        return {}
    
    print(f"\n🔍 加载 Cluster-Level GT: {gt_path}")
    
    gt_dict = {}
    with open(gt_path, 'r') as f:
        header = f.readline()  # 跳过表头
        print(f"   表头: {header.strip()}")
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    cluster_id = int(parts[0])
                    ref_seq = parts[1]
                    gt_dict[cluster_id] = ref_seq
                    
                    if len(gt_dict) <= 3:
                        print(f"     GT[{cluster_id}]: {ref_seq[:50]}...")
                except ValueError:
                    continue
    
    print(f"   ✅ 加载 {len(gt_dict)} 个 Cluster GT 序列")
    if gt_dict:
        print(f"   GT簇ID范围: {min(gt_dict.keys())} - {max(gt_dict.keys())}")
    
    return gt_dict


def build_sequence_to_gt_mapping(feddna_reads: List[str], 
                                 raw_reads: Dict[str, str],
                                 read_gt: Dict[str, Tuple[int, str, str]]) -> List[int]:
    """
    建立 FedDNA reads 到原始 GT 簇的映射
    
    通过序列匹配找到每个read对应的原始GT簇ID
    """
    # 反向映射：sequence -> read_id
    seq_to_id = {seq: rid for rid, seq in raw_reads.items()}
    
    original_gt_labels = []
    matched = 0
    
    for seq in feddna_reads:
        if seq in seq_to_id:
            read_id = seq_to_id[seq]
            if read_id in read_gt:
                gt_cluster_id = read_gt[read_id][0]
                original_gt_labels.append(gt_cluster_id)
                matched += 1
            else:
                original_gt_labels.append(-1)
        else:
            original_gt_labels.append(-1)
    
    print(f"   ✅ GT标签匹配: {matched}/{len(feddna_reads)} ({matched/len(feddna_reads)*100:.1f}%)")
    
    return original_gt_labels


# ============================================================================
# 数据管理器 - 适配版
# ============================================================================
class ClusterDataManager:
    """簇数据管理器 - 适配你的数据格式"""
    
    def __init__(self, experiment_dir: str, config: Config):
        self.experiment_dir = experiment_dir
        self.config = config
        
        # 路径
        self.raw_dir = os.path.join(experiment_dir, "01_RawData")
        self.feddna_dir = os.path.join(experiment_dir, "03_FedDNA_In")
        
        # 数据存储
        self.reads: List[str] = []
        self.qualities: List[str] = []
        self.clover_labels: np.ndarray = None  # Clover聚类结果
        self.original_gt_labels: np.ndarray = None  # 原始GT簇标签
        self.current_labels: np.ndarray = None
        
        # 簇管理
        self.cluster_assignments: Dict[int, Set[int]] = defaultdict(set)
        self.cluster_status: Dict[int, str] = {}
        self.noise_reads: Set[int] = set()
        
        # GT
        self.cluster_gt: Dict[int, str] = {}  # 簇级GT
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载所有数据"""
        print("\n" + "=" * 60)
        print("📂 加载数据")
        print("=" * 60)
        
        # 1. 加载 FedDNA 格式的 reads
        self.reads, clover_labels = load_feddna_format(self.feddna_dir)
        self.clover_labels = np.array(clover_labels)
        
        # 2. 生成默认质量分数
        self.qualities = ['I' * len(read) for read in self.reads]
        print(f"   ✅ 生成默认质量分数")
        
        # 3. 加载原始reads和GT
        raw_reads = load_raw_reads_with_ids(self.raw_dir)
        read_gt = load_read_level_gt(self.raw_dir)
        self.cluster_gt = load_cluster_level_gt(self.raw_dir)
        
        # 4. 建立GT映射
        if raw_reads and read_gt:
            self.original_gt_labels = np.array(
                build_sequence_to_gt_mapping(self.reads, raw_reads, read_gt)
            )
        else:
            self.original_gt_labels = np.full(len(self.reads), -1)
            print(f"   ⚠️ 无法建立GT映射，使用-1填充")
        
        # 5. 初始化当前标签 (使用Clover结果作为起点)
        self.current_labels = self.clover_labels.copy()
        
        # 6. 初始化簇分配
        for idx, label in enumerate(self.current_labels):
            if label >= 0:
                self.cluster_assignments[label].add(idx)
                self.cluster_status[label] = 'healthy'
        
        self.total_reads = len(self.reads)
        
        print(f"\n📊 数据摘要:")
        print(f"   - 总Reads: {self.total_reads}")
        print(f"   - Clover簇数: {len(np.unique(self.clover_labels))}")
        print(f"   - GT簇数: {len(self.cluster_gt)}")
        print(f"   - 目标K: {self.config.k_target}")
    
    def get_cluster_reads(self, cluster_id: int) -> List[int]:
        """获取簇内所有read索引"""
        return list(self.cluster_assignments.get(cluster_id, set()))
    
    def get_active_clusters(self) -> List[int]:
        """获取所有活跃簇ID"""
        return [cid for cid, reads in self.cluster_assignments.items() 
                if len(reads) > 0 and self.cluster_status.get(cid) != 'eliminated']
    
    def reassign_read(self, read_idx: int, new_cluster_id: int):
        """重分配read到新簇"""
        old_cluster = self.current_labels[read_idx]
        
        if old_cluster >= 0 and old_cluster in self.cluster_assignments:
            self.cluster_assignments[old_cluster].discard(read_idx)
        
        self.noise_reads.discard(read_idx)
        
        self.current_labels[read_idx] = new_cluster_id
        self.cluster_assignments[new_cluster_id].add(read_idx)
        
        if new_cluster_id not in self.cluster_status:
            self.cluster_status[new_cluster_id] = 'healthy'
    
    def mark_as_noise(self, read_idx: int):
        """标记read为噪声"""
        old_cluster = self.current_labels[read_idx]
        
        if old_cluster >= 0 and old_cluster in self.cluster_assignments:
            self.cluster_assignments[old_cluster].discard(read_idx)
        
        self.current_labels[read_idx] = -1
        self.noise_reads.add(read_idx)
    
    def remove_cluster(self, cluster_id: int):
        """移除簇"""
        if cluster_id in self.cluster_assignments:
            del self.cluster_assignments[cluster_id]
        self.cluster_status[cluster_id] = 'eliminated'
    
    def get_k_effective(self) -> int:
        """获取当前有效簇数"""
        return len([cid for cid, reads in self.cluster_assignments.items() 
                   if len(reads) > 0])


# ============================================================================
# 序列编码
# ============================================================================
class SequenceEncoder:
    """序列编码器"""
    
    BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    
    @staticmethod
    def encode_sequence(seq: str, quality: str, max_len: int = 150) -> torch.Tensor:
        """编码序列为张量"""
        seq_len = min(len(seq), max_len)
        encoding = torch.zeros(max_len, 6)
        
        for i in range(seq_len):
            base = seq[i].upper()
            if base in SequenceEncoder.BASE_MAP and SequenceEncoder.BASE_MAP[base] < 4:
                encoding[i, SequenceEncoder.BASE_MAP[base]] = 1.0
            
            if i < len(quality):
                q = ord(quality[i]) - 33
                encoding[i, 4] = q / 40.0
            else:
                encoding[i, 4] = 0.5
            
            encoding[i, 5] = i / max_len
        
        return encoding


# ============================================================================
# 数据集
# ============================================================================
class ClusterDataset(Dataset):
    """簇数据集"""
    
    def __init__(self, data_manager: ClusterDataManager, max_len: int = 150):
        self.data_manager = data_manager
        self.max_len = max_len
        self.encoder = SequenceEncoder()
        
        self.valid_indices = [i for i in range(len(data_manager.reads))
                             if i not in data_manager.noise_reads]
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        seq = self.data_manager.reads[real_idx]
        qual = self.data_manager.qualities[real_idx]
        label = self.data_manager.current_labels[real_idx]
        
        encoding = self.encoder.encode_sequence(seq, qual, self.max_len)
        
        return {
            'encoding': encoding,
            'label': label,
            'index': real_idx
        }


# ============================================================================
# 模型
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_encoding = PositionalEncoding(config.hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_proj = nn.Linear(config.hidden_dim, config.latent_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return F.normalize(x, p=2, dim=-1)


class SequenceDecoder(nn.Module):
    def __init__(self, config: Config, max_len: int = 150):
        super().__init__()
        self.max_len = max_len
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, max_len * 4)
        )
    
    def forward(self, z):
        x = self.latent_proj(z)
        x = self.decoder(x)
        x = x.view(-1, self.max_len, 4)
        return x


class ClusterReconstructionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = SequenceDecoder(config)
        self.cluster_centers = nn.Parameter(
            torch.randn(config.k_target + 20, config.latent_dim)
        )
        nn.init.xavier_uniform_(self.cluster_centers)
    
    def forward(self, x):
        z = self.encoder(x)
        logits = self.decoder(z)
        return z, logits


# ============================================================================
# 损失函数
# ============================================================================
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)
        
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float()
        positive_mask.fill_diagonal_(0)
        
        diag_mask = torch.eye(batch_size, device=z.device)
        exp_sim = torch.exp(sim_matrix) * (1 - diag_mask)
        
        positive_sum = (exp_sim * positive_mask).sum(dim=1)
        total_sum = exp_sim.sum(dim=1)
        
        valid_mask = positive_sum > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        loss = -torch.log(positive_sum[valid_mask] / (total_sum[valid_mask] + 1e-8))
        return loss.mean()


class DELLoss(nn.Module):
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        unique_labels = torch.unique(labels[labels >= 0])
        
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=z.device)
        
        centers = []
        intra_vars = []
        
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                cluster_z = z[mask]
                center = cluster_z.mean(dim=0)
                centers.append(center)
                
                if cluster_z.size(0) > 1:
                    var = ((cluster_z - center) ** 2).sum(dim=1).mean()
                    intra_vars.append(var)
        
        if len(centers) < 2:
            return torch.tensor(0.0, device=z.device)
        
        centers = torch.stack(centers)
        
        inter_dist = torch.cdist(centers, centers)
        inter_dist = inter_dist[torch.triu(torch.ones_like(inter_dist), diagonal=1) == 1]
        
        intra_loss = torch.stack(intra_vars).mean() if intra_vars else torch.tensor(0.0, device=z.device)
        inter_loss = -inter_dist.mean() if inter_dist.numel() > 0 else torch.tensor(0.0, device=z.device)
        
        return intra_loss + 0.5 * inter_loss


class StrictKConstraintLoss(nn.Module):
    def __init__(self, k_target: int, lambda_k: float = 2.0):
        super().__init__()
        self.k_target = k_target
        self.lambda_k = lambda_k
    
    def forward(self, k_effective: int, epoch: int, max_epochs: int) -> torch.Tensor:
        diff = k_effective - self.k_target
        tolerance = 2
        
        if abs(diff) <= tolerance:
            penalty = (diff ** 2) * 0.1
        else:
            excess = abs(diff) - tolerance
            penalty = tolerance ** 2 * 0.1 + excess ** 3
        
        epoch_factor = 1.0 + (epoch / max_epochs) * 2.0
        
        return torch.tensor(self.lambda_k * penalty * epoch_factor, dtype=torch.float32)


# ============================================================================
# 辅助函数
# ============================================================================
def calculate_sequence_similarity(seq1: str, seq2: str) -> float:
    """计算两个序列的相似度"""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0
    matches = sum(c1 == c2 for c1, c2 in zip(seq1[:min_len], seq2[:min_len]))
    return matches / min_len


def calculate_cluster_consistency(reads: List[str]) -> float:
    """计算簇内一致性"""
    if len(reads) < 2:
        return 1.0
    
    sample_size = min(20, len(reads))
    sampled_reads = list(np.random.choice(reads, sample_size, replace=False)) if len(reads) > sample_size else reads
    
    max_len = max(len(r) for r in sampled_reads)
    consensus = []
    
    for pos in range(max_len):
        bases = [read[pos] for read in sampled_reads if pos < len(read)]
        if bases:
            base_counts = Counter(bases)
            consensus.append(base_counts.most_common(1)[0][0])
    
    consensus_seq = ''.join(consensus)
    
    consistencies = [calculate_sequence_similarity(read, consensus_seq) for read in sampled_reads]
    return np.mean(consistencies)


def build_consensus_sequence(reads: List[str]) -> str:
    """构建consensus序列"""
    if not reads:
        return ""
    if len(reads) == 1:
        return reads[0]
    
    max_len = max(len(r) for r in reads)
    consensus = []
    
    for pos in range(max_len):
        bases = [read[pos] for read in reads if pos < len(read)]
        if bases:
            base_counts = Counter(bases)
            consensus.append(base_counts.most_common(1)[0][0])
    
    return ''.join(consensus)


def find_nearest_healthy_cluster(data_manager: ClusterDataManager, 
                                read_idx: int, 
                                exclude_cids: Set[int] = None,
                                min_cluster_size: int = 5) -> Optional[int]:
    """找到最近的健康簇"""
    exclude_cids = exclude_cids or set()
    
    read_seq = data_manager.reads[read_idx]
    best_cid = None
    best_similarity = 0.0
    
    for cid in data_manager.cluster_assignments.keys():
        if cid in exclude_cids:
            continue
        if data_manager.cluster_status.get(cid) == 'eliminated':
            continue
        
        cluster_reads = data_manager.get_cluster_reads(cid)
        if len(cluster_reads) < min_cluster_size:
            continue
        
        sample_indices = cluster_reads[:min(5, len(cluster_reads))]
        similarities = []
        
        for center_idx in sample_indices:
            center_seq = data_manager.reads[center_idx]
            sim = calculate_sequence_similarity(read_seq, center_seq)
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        
        if avg_similarity > best_similarity and avg_similarity > 0.65:
            best_similarity = avg_similarity
            best_cid = cid
    
    return best_cid


# ============================================================================
# 簇健康度评估与淘汰
# ============================================================================
def evaluate_cluster_health(data_manager: ClusterDataManager, 
                           config: Config) -> Tuple[List[int], List[int]]:
    """评估簇健康度"""
    healthy_clusters = []
    weak_clusters = []
    
    min_cluster_size = max(5, int(data_manager.total_reads * config.min_cluster_ratio))
    
    for cid in data_manager.get_active_clusters():
        cluster_reads = data_manager.get_cluster_reads(cid)
        cluster_size = len(cluster_reads)
        
        is_weak = False
        
        if cluster_size < min_cluster_size:
            is_weak = True
        
        if not is_weak and cluster_size >= 3:
            reads = [data_manager.reads[idx] for idx in cluster_reads[:20]]
            consistency = calculate_cluster_consistency(reads)
            if consistency < config.weak_consistency_threshold:
                is_weak = True
        
        if is_weak:
            weak_clusters.append(cid)
            data_manager.cluster_status[cid] = 'weak'
        else:
            healthy_clusters.append(cid)
            data_manager.cluster_status[cid] = 'healthy'
    
    return healthy_clusters, weak_clusters


def eliminate_weak_clusters(data_manager: ClusterDataManager,
                           config: Config,
                           epoch: int,
                           max_epochs: int) -> int:
    """淘汰弱簇"""
    eliminated = 0
    
    progress = epoch / max_epochs
    min_cluster_size = max(5, int(data_manager.total_reads * config.min_cluster_ratio))
    
    if progress > 0.7:
        min_cluster_size = max(10, int(data_manager.total_reads * 0.01))
    
    weak_clusters_info = []
    
    for cid in list(data_manager.cluster_assignments.keys()):
        if data_manager.cluster_status.get(cid) == 'eliminated':
            continue
        
        cluster_reads = data_manager.get_cluster_reads(cid)
        cluster_size = len(cluster_reads)
        
        is_weak = False
        reason = ""
        
        if cluster_size < min_cluster_size and cluster_size > 0:
            is_weak = True
            reason = f"size={cluster_size}<{min_cluster_size}"
        
        if not is_weak and cluster_size >= 3:
            reads = [data_manager.reads[idx] for idx in cluster_reads[:15]]
            consistency = calculate_cluster_consistency(reads)
            if consistency < config.weak_consistency_threshold:
                is_weak = True
                reason = f"consistency={consistency:.1%}"
        
        if is_weak:
            weak_clusters_info.append((cid, cluster_size, reason))
    
    weak_clusters_info.sort(key=lambda x: x[1])
    
    if weak_clusters_info:
        print(f"   发现 {len(weak_clusters_info)} 个弱簇待淘汰")
    
    for cid, size, reason in weak_clusters_info:
        cluster_reads = data_manager.get_cluster_reads(cid)
        
        reassigned = 0
        marked_noise = 0
        
        for read_idx in list(cluster_reads):
            best_cid = find_nearest_healthy_cluster(
                data_manager, read_idx, exclude_cids={cid}, min_cluster_size=min_cluster_size
            )
            
            if best_cid is not None:
                data_manager.reassign_read(read_idx, best_cid)
                reassigned += 1
            else:
                data_manager.mark_as_noise(read_idx)
                marked_noise += 1
        
        data_manager.remove_cluster(cid)
        eliminated += 1
        
        print(f"   ❌ 淘汰簇{cid}: {reason}, 重分配{reassigned}, 噪声{marked_noise}")
    
    return eliminated


# ============================================================================
# 困难样本挖掘
# ============================================================================
def mine_hard_samples(data_manager: ClusterDataManager,
                     model: ClusterReconstructionModel,
                     config: Config) -> Tuple[int, int]:
    """困难样本挖掘"""
    device = config.device
    model.eval()
    
    reassigned = 0
    new_noise = 0
    
    encoder = SequenceEncoder()
    
    cluster_centers = {}
    for cid in data_manager.get_active_clusters():
        cluster_reads = data_manager.get_cluster_reads(cid)
        if len(cluster_reads) < 3:
            continue
        
        sample_indices = cluster_reads[:min(20, len(cluster_reads))]
        encodings = []
        
        for idx in sample_indices:
            enc = encoder.encode_sequence(
                data_manager.reads[idx], 
                data_manager.qualities[idx]
            )
            encodings.append(enc)
        
        encodings = torch.stack(encodings).to(device)
        
        with torch.no_grad():
            z, _ = model(encodings)
            center = z.mean(dim=0)
            cluster_centers[cid] = center
    
    if not cluster_centers:
        return 0, 0
    
    for idx in range(data_manager.total_reads):
        if idx in data_manager.noise_reads:
            continue
        
        current_label = data_manager.current_labels[idx]
        if current_label < 0 or current_label not in cluster_centers:
            continue
        
        enc = encoder.encode_sequence(
            data_manager.reads[idx],
            data_manager.qualities[idx]
        ).unsqueeze(0).to(device)
        
        with torch.no_grad():
            z, _ = model(enc)
            z = z.squeeze(0)
        
        current_center = cluster_centers[current_label]
        current_sim = F.cosine_similarity(z.unsqueeze(0), current_center.unsqueeze(0)).item()
        
        if current_sim < 0.5:
            best_cid = None
            best_sim = current_sim
            
            for cid, center in cluster_centers.items():
                if cid == current_label:
                    continue
                
                sim = F.cosine_similarity(z.unsqueeze(0), center.unsqueeze(0)).item()
                if sim > best_sim + 0.1:
                    best_sim = sim
                    best_cid = cid
            
            if best_cid is not None and best_sim > 0.6:
                data_manager.reassign_read(idx, best_cid)
                reassigned += 1
            elif current_sim < 0.3:
                data_manager.mark_as_noise(idx)
                new_noise += 1
    
    model.train()
    return reassigned, new_noise


# ============================================================================
# 序列重建
# ============================================================================
def reconstruct_sequences(data_manager: ClusterDataManager,
                         model: ClusterReconstructionModel,
                         config: Config) -> Dict[int, str]:
    """为每个簇重建参考序列"""
    model.eval()
    
    reconstructed = {}
    
    print("\n🧬 序列重建...")
    
    for cid in sorted(data_manager.get_active_clusters()):
        cluster_reads = data_manager.get_cluster_reads(cid)
        
        if len(cluster_reads) == 0:
            continue
        
        reads = [data_manager.reads[idx] for idx in cluster_reads]
        consensus = build_consensus_sequence(reads)
        reconstructed[cid] = consensus
        
        print(f"   簇{cid:>2} ({len(cluster_reads):>3} reads): {consensus[:50]}...")
    
    model.train()
    return reconstructed


# ============================================================================
# 验证
# ============================================================================
def validate_results(reconstructed: Dict[int, str], 
                    data_manager: ClusterDataManager) -> Dict:
    """验证结果 - 使用已加载的GT"""
    
    results = {
        'cluster_info': [],
        'avg_consistency': 0.0,
        'avg_gt_accuracy': 0.0,
        'total_clusters': 0,
        'gt_matched_clusters': 0
    }
    
    cluster_gt = data_manager.cluster_gt
    
    print("\n" + "=" * 90)
    print("📊 验证结果")
    print("=" * 90)
    print(f"{'簇ID':>6} | {'Reads':>6} | {'一致性':>10} | {'GT准确率':>10} | {'匹配GT簇':>8} | {'状态':>8}")
    print("-" * 90)
    
    consistency_scores = []
    gt_accuracy_scores = []
    
    for cid in sorted(reconstructed.keys()):
        recon_seq = reconstructed[cid]
        read_indices = data_manager.get_cluster_reads(cid)
        num_reads = len(read_indices)
        status = data_manager.cluster_status.get(cid, 'unknown')
        
        # 1. 簇内一致性
        reads = [data_manager.reads[idx] for idx in read_indices]
        avg_consistency = calculate_cluster_consistency(reads) if reads else 0.0
        consistency_scores.append(avg_consistency)
        
        # 2. GT准确率 - 通过原始GT标签找到对应的GT序列
        gt_accuracy = None
        matched_gt_cid = None
        
        if cluster_gt:
            # 找到该簇中reads的原始GT标签
            original_labels = [data_manager.original_gt_labels[idx] for idx in read_indices
                             if data_manager.original_gt_labels[idx] >= 0]
            
            if original_labels:
                # 多数投票
                label_counts = Counter(original_labels)
                most_common_label, count = label_counts.most_common(1)[0]
                
                if most_common_label in cluster_gt:
                    gt_seq = cluster_gt[most_common_label]
                    gt_accuracy = calculate_sequence_similarity(recon_seq, gt_seq)
                    gt_accuracy_scores.append(gt_accuracy)
                    matched_gt_cid = most_common_label
        
        # 状态显示
        status_str = '✓ 健康' if status == 'healthy' else ('⚠ 弱' if status == 'weak' else '❓')
        gt_str = f"{gt_accuracy*100:>8.1f}%" if gt_accuracy is not None else "      N/A"
        gt_cid_str = f"{matched_gt_cid:>8}" if matched_gt_cid is not None else "     N/A"
        
        print(f"{cid:>6} | {num_reads:>6} | {avg_consistency*100:>9.1f}% | {gt_str:>10} | {gt_cid_str:>8} | {status_str:>8}")
        
        results['cluster_info'].append({
            'cluster_id': cid,
            'num_reads': num_reads,
            'consistency': avg_consistency,
            'gt_accuracy': gt_accuracy,
            'matched_gt_cid': matched_gt_cid,
            'status': status,
            'sequence': recon_seq
        })
    
    print("-" * 90)
    
    # 汇总
    results['avg_consistency'] = np.mean(consistency_scores) if consistency_scores else 0.0
    results['avg_gt_accuracy'] = np.mean(gt_accuracy_scores) if gt_accuracy_scores else 0.0
    results['total_clusters'] = len(reconstructed)
    results['gt_matched_clusters'] = len(gt_accuracy_scores)
    
    print(f"\n📈 汇总:")
    print(f"   - 总簇数: {results['total_clusters']} (目标: {data_manager.config.k_target})")
    print(f"   - 平均Read一致性: {results['avg_consistency']*100:.2f}%")
    
    if gt_accuracy_scores:
        print(f"   - 平均GT准确率: {results['avg_gt_accuracy']*100:.2f}%")
        print(f"   - GT验证覆盖: {results['gt_matched_clusters']}/{results['total_clusters']} 簇")
        
        # 分级统计
        excellent = sum(1 for acc in gt_accuracy_scores if acc >= 0.95)
        good = sum(1 for acc in gt_accuracy_scores if 0.9 <= acc < 0.95)
        fair = sum(1 for acc in gt_accuracy_scores if 0.8 <= acc < 0.9)
        poor = sum(1 for acc in gt_accuracy_scores if acc < 0.8)
        
        print(f"   - GT准确率分布:")
        print(f"     ≥95%: {excellent} ({excellent/len(gt_accuracy_scores)*100:.1f}%)")
        print(f"     90-95%: {good} ({good/len(gt_accuracy_scores)*100:.1f}%)")
        print(f"     80-90%: {fair} ({fair/len(gt_accuracy_scores)*100:.1f}%)")
        print(f"     <80%: {poor} ({poor/len(gt_accuracy_scores)*100:.1f}%)")
    
    noise_ratio = len(data_manager.noise_reads) / data_manager.total_reads * 100
    print(f"   - 噪声Reads: {len(data_manager.noise_reads)} ({noise_ratio:.1f}%)")
    
    return results


# ============================================================================
# 保存结果
# ============================================================================
def save_results(reconstructed: Dict[int, str],
                data_manager: ClusterDataManager,
                results: Dict,
                output_dir: str,
                training_history: Dict = None):
    """保存所有结果"""
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n💾 保存结果到: {output_dir}")
    
    # 1. FASTA
    fasta_path = os.path.join(output_dir, "reconstructed_sequences.fasta")
    with open(fasta_path, 'w') as f:
        for cid in sorted(reconstructed.keys()):
            seq = reconstructed[cid]
            num_reads = len(data_manager.get_cluster_reads(cid))
            status = data_manager.cluster_status.get(cid, 'unknown')
            f.write(f">cluster_{cid}_reads_{num_reads}_status_{status}\n")
            f.write(f"{seq}\n")
    print(f"   ✅ 序列: reconstructed_sequences.fasta")
    
    # 2. 纯序列
    ref_path = os.path.join(output_dir, "ref.txt")
    with open(ref_path, 'w') as f:
        for cid in sorted(reconstructed.keys()):
            f.write(f"{reconstructed[cid]}\n")
    print(f"   ✅ 纯序列: ref.txt")
    
    # 3. 簇分配
    assign_path = os.path.join(output_dir, "cluster_assignments.txt")
    with open(assign_path, 'w') as f:
        f.write("Read_Index\tCluster_ID\tOriginal_GT_Cluster\n")
        for idx in range(data_manager.total_reads):
            label = data_manager.current_labels[idx]
            gt_label = data_manager.original_gt_labels[idx]
            f.write(f"{idx}\t{label}\t{gt_label}\n")
    print(f"   ✅ 分配: cluster_assignments.txt")
    
    # 4. 簇健康度
    health_path = os.path.join(output_dir, "cluster_health.txt")
    with open(health_path, 'w') as f:
        f.write("Cluster_ID\tNum_Reads\tConsistency\tGT_Accuracy\tMatched_GT_Cluster\tStatus\n")
        for info in results['cluster_info']:
            gt_acc = info['gt_accuracy'] if info['gt_accuracy'] is not None else -1
            gt_cid = info['matched_gt_cid'] if info['matched_gt_cid'] is not None else -1
            f.write(f"{info['cluster_id']}\t{info['num_reads']}\t")
            f.write(f"{info['consistency']:.4f}\t{gt_acc:.4f}\t{gt_cid}\t{info['status']}\n")
    print(f"   ✅ 健康度: cluster_health.txt")
    
    # 5. 训练历史
    if training_history:
        history_path = os.path.join(output_dir, "training_history.txt")
        with open(history_path, 'w') as f:
            f.write("Epoch\tTotal_Loss\tContrastive\tDEL\tK_Constraint\tK_Effective\n")
            for i in range(len(training_history['total_loss'])):
                f.write(f"{i+1}\t{training_history['total_loss'][i]:.4f}\t")
                f.write(f"{training_history['contrastive_loss'][i]:.4f}\t")
                f.write(f"{training_history['del_loss'][i]:.4f}\t")
                f.write(f"{training_history['k_loss'][i]:.4f}\t")
                f.write(f"{training_history['k_effective'][i]}\n")
        print(f"   ✅ 历史: training_history.txt")
        
        # 绘图
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            epochs = range(1, len(training_history['total_loss']) + 1)
            
            axes[0, 0].plot(epochs, training_history['total_loss'], 'b-')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(epochs, training_history['contrastive_loss'], 'r-', label='Contrastive')
            axes[0, 1].plot(epochs, training_history['del_loss'], 'g-', label='DEL')
            axes[0, 1].set_title('Loss Components')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            axes[1, 0].plot(epochs, training_history['k_loss'], 'm-')
            axes[1, 0].set_title('K Constraint Loss')
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(epochs, training_history['k_effective'], 'c-')
            axes[1, 1].axhline(y=data_manager.config.k_target, color='r', linestyle='--', label=f'Target K={data_manager.config.k_target}')
            axes[1, 1].set_title('Effective Cluster Count')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)
            plt.close()
            print(f"   ✅ 训练曲线: training_history.png")
        except Exception as e:
            print(f"   ⚠️ 绘图失败: {e}")
    
    # 6. 噪声
    noise_path = os.path.join(output_dir, "noise_reads.txt")
    with open(noise_path, 'w') as f:
        f.write(f"# Total: {len(data_manager.noise_reads)}\n")
        for idx in sorted(data_manager.noise_reads):
            f.write(f"{idx}\n")
    print(f"   ✅ 噪声: noise_reads.txt ({len(data_manager.noise_reads)} reads)")


# ============================================================================
# 训练
# ============================================================================
def train(data_manager: ClusterDataManager,
         model: ClusterReconstructionModel,
         config: Config) -> Dict:
    """训练主循环"""
    
    device = config.device
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    contrastive_loss_fn = ContrastiveLoss()
    del_loss_fn = DELLoss()
    k_constraint_fn = StrictKConstraintLoss(config.k_target, config.lambda_k)
    
    history = {
        'total_loss': [],
        'contrastive_loss': [],
        'del_loss': [],
        'k_loss': [],
        'k_effective': []
    }
    
    print("\n" + "=" * 70)
    print("🚀 开始训练")
    print("=" * 70)
    
    for epoch in range(1, config.num_epochs + 1):
        dataset = ClusterDataset(data_manager)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        epoch_losses = {'total': 0, 'contrastive': 0, 'del': 0, 'k': 0}
        num_batches = 0
        
        model.train()
        
        for batch in dataloader:
            encodings = batch['encoding'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            z, logits = model(encodings)
            
            loss_contrastive = contrastive_loss_fn(z, labels)
            loss_del = del_loss_fn(z, labels)
            
            total_loss = config.lambda_contrastive * loss_contrastive + config.lambda_del * loss_del
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses['total'] += total_loss.item()
            epoch_losses['contrastive'] += loss_contrastive.item()
            epoch_losses['del'] += loss_del.item()
            num_batches += 1
        
        scheduler.step()
        
        k_effective = data_manager.get_k_effective()
        loss_k = k_constraint_fn(k_effective, epoch, config.num_epochs)
        epoch_losses['k'] = loss_k.item()
        
        avg_total = epoch_losses['total'] / num_batches if num_batches > 0 else 0
        avg_contrastive = epoch_losses['contrastive'] / num_batches if num_batches > 0 else 0
        avg_del = epoch_losses['del'] / num_batches if num_batches > 0 else 0
        
        history['total_loss'].append(avg_total + epoch_losses['k'])
        history['contrastive_loss'].append(avg_contrastive)
        history['del_loss'].append(avg_del)
        history['k_loss'].append(epoch_losses['k'])
        history['k_effective'].append(k_effective)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"\n{'='*70}")
            print(f"📍 Epoch {epoch}/{config.num_epochs}")
            print(f"{'='*70}")
            print(f"\n📈 损失:")
            print(f"   - Total: {avg_total + epoch_losses['k']:.4f}")
            print(f"   - Contrastive: {avg_contrastive:.4f}")
            print(f"   - DEL: {avg_del:.4f}")
            print(f"   - K-constraint: {epoch_losses['k']:.4f}")
        
        if epoch % 5 == 0:
            if epoch % 10 == 0:
                print(f"\n🏥 簇健康度评估...")
            healthy, weak = evaluate_cluster_health(data_manager, config)
            if epoch % 10 == 0:
                print(f"   - 健康簇: {len(healthy)}")
                print(f"   - 弱簇: {len(weak)}")
        
        if epoch % 10 == 0 and epoch > 10:
            print(f"\n🔧 困难样本修正...")
            reassigned, new_noise = mine_hard_samples(data_manager, model, config)
            print(f"   - 重分配: {reassigned}")
            print(f"   - 新噪声: {new_noise}")
        
        eliminate_freq = 10 if epoch < config.num_epochs * 0.7 else 5
        if epoch % eliminate_freq == 0 and epoch > 10:
            if epoch % 10 == 0:
                print(f"\n🗑️ 弱簇淘汰检查...")
            eliminated = eliminate_weak_clusters(data_manager, config, epoch, config.num_epochs)
            if eliminated > 0 and epoch % 10 != 0:
                print(f"   淘汰了 {eliminated} 个弱簇")
        
        if epoch % 10 == 0:
            k_eff = data_manager.get_k_effective()
            noise_count = len(data_manager.noise_reads)
            noise_ratio = noise_count / data_manager.total_reads * 100
            
            print(f"\n📊 当前状态:")
            print(f"   - K_effective: {k_eff} (目标: {config.k_target})")
            print(f"   - 噪声率: {noise_ratio:.1f}%")
    
    print("\n" + "=" * 70)
    print("🎉 训练完成!")
    print("=" * 70)
    
    k_final = data_manager.get_k_effective()
    healthy_final, weak_final = evaluate_cluster_health(data_manager, config)
    valid_reads = data_manager.total_reads - len(data_manager.noise_reads)
    avg_cluster_size = valid_reads / k_final if k_final > 0 else 0
    
    print(f"\n📊 最终状态:")
    print(f"   - K_effective: {k_final}")
    print(f"   - K_healthy: {len(healthy_final)}")
    print(f"   - 有效Reads: {valid_reads}")
    print(f"   - 噪声Reads: {len(data_manager.noise_reads)} ({len(data_manager.noise_reads)/data_manager.total_reads*100:.1f}%)")
    print(f"   - 平均簇大小: {avg_cluster_size:.1f}")
    
    return history


# ============================================================================
# 主函数
# ============================================================================
def main():
    """主函数"""
    
    # ==========================================
    # 配置路径 - 修改这里！
    # ==========================================
    # 实验目录 (包含 01_RawData, 02_CloverOut, 03_FedDNA_In)
    EXPERIMENT_DIR = "CC/Step0/Experiments/20251217_015615_Cluster_GT_Test"
    
    # 输出目录
    OUTPUT_DIR = f"./results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # ==========================================
    # 配置参数
    # ==========================================
    config = Config(
        experiment_dir=EXPERIMENT_DIR,
        
        # 模型
        hidden_dim=128,
        latent_dim=64,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        
        # 训练
        k_target=50,
        batch_size=64,
        num_epochs=50,
        learning_rate=1e-3,
        
        # 损失
        lambda_contrastive=1.0,
        lambda_del=0.5,
        lambda_k=2.0,
        
        # 阈值
        similarity_threshold=0.7,
        min_cluster_ratio=0.005,
        weak_consistency_threshold=0.6,
    )
    
    print("=" * 70)
    print("🧬 FedDNA 簇序列重建系统 v3 (适配版)")
    print("=" * 70)
    print(f"📂 实验目录: {EXPERIMENT_DIR}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"🎯 目标簇数K: {config.k_target}")
    print(f"🖥️ 设备: {config.device}")
    print("=" * 70)
    
    # ==========================================
    # 检查目录
    # ==========================================
    if not os.path.exists(EXPERIMENT_DIR):
        print(f"❌ 实验目录不存在: {EXPERIMENT_DIR}")
        print(f"💡 请确保路径正确，或先运行数据生成脚本")
        return
    
    feddna_dir = os.path.join(EXPERIMENT_DIR, "03_FedDNA_In")
    if not os.path.exists(feddna_dir):
        print(f"❌ FedDNA数据目录不存在: {feddna_dir}")
        return
    
    # ==========================================
    # 加载数据
    # ==========================================
    try:
        data_manager = ClusterDataManager(EXPERIMENT_DIR, config)
    except FileNotFoundError as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # ==========================================
    # 创建模型
    # ==========================================
    model = ClusterReconstructionModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 模型参数: {total_params:,} (可训练: {trainable_params:,})")
    
    # ==========================================
    # 训练
    # ==========================================
    training_history = train(data_manager, model, config)
    
    # ==========================================
    # 序列重建
    # ==========================================
    reconstructed = reconstruct_sequences(data_manager, model, config)
    
    # ==========================================
    # 验证结果
    # ==========================================
    results = validate_results(reconstructed, data_manager)
    
    # ==========================================
    # 保存结果
    # ==========================================
    save_results(reconstructed, data_manager, results, OUTPUT_DIR, training_history)
    
    # 保存模型
    model_path = os.path.join(OUTPUT_DIR, "model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'k_effective': data_manager.get_k_effective(),
    }, model_path)
    print(f"   ✅ 模型: model.pth")
    
    print("\n" + "=" * 70)
    print("🎉 全部完成!")
    print("=" * 70)
    
    return results


# ============================================================================
# 命令行接口
# ============================================================================
def parse_args():
    """解析命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FedDNA 簇序列重建系统 v3')
    
    parser.add_argument('--experiment_dir', type=str, required=False,
                       help='实验目录路径 (包含01_RawData, 03_FedDNA_In等)')
    parser.add_argument('--output_dir', type=str, required=False,
                       help='输出目录')
    parser.add_argument('--k_target', type=int, default=50,
                       help='目标簇数K')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--lambda_k', type=float, default=2.0,
                       help='K约束权重')
    
    return parser.parse_args()


if __name__ == "__main__":
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        args = parse_args()
        
        if args.experiment_dir:
            # 更新主函数中的路径
            # 这里简单处理：直接修改全局变量或重新调用
            config = Config(
                experiment_dir=args.experiment_dir,
                k_target=args.k_target,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                lambda_k=args.lambda_k,
            )
            
            output_dir = args.output_dir or f"./results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            print("=" * 70)
            print("🧬 FedDNA 簇序列重建系统 v3 (命令行模式)")
            print("=" * 70)
            
            # 加载数据
            data_manager = ClusterDataManager(args.experiment_dir, config)
            
            # 创建模型
            model = ClusterReconstructionModel(config)
            
            # 训练
            training_history = train(data_manager, model, config)
            
            # 重建
            reconstructed = reconstruct_sequences(data_manager, model, config)
            
            # 验证
            results = validate_results(reconstructed, data_manager)
            
            # 保存
            save_results(reconstructed, data_manager, results, output_dir, training_history)
            
            print("\n🎉 完成!")
        else:
            print("请提供 --experiment_dir 参数")
    else:
        # 直接运行
        main()

