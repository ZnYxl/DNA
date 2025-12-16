#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧬 证据驱动的DNA聚类优化模型 (Evidence-Driven Clustering)

核心思想:
1. 不纠正代表序列，而是优化reads的簇分配
2. 基于证据强度识别困难样本
3. 渐进式优化，稳定收敛

簇数量策略:
- 初始K来自Clover
- 训练中允许: 删除空簇、丢弃噪声
- 保持K相对稳定
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 导入FedDNA核心组件
# ==========================================
try:
    from models.conmamba import ConmambaBlock
    print("✅ 成功导入 ConmambaBlock")
except ImportError as e:
    print(f"⚠️ 导入失败: {e}, 使用简化版本")
    ConmambaBlock = None

# ==========================================
# 工具函数
# ==========================================

def safe_log(x, eps=1e-8):
    return torch.log(torch.clamp(x, min=eps))

def safe_div(x, y, eps=1e-8):
    return x / torch.clamp(y, min=eps)

# ==========================================
# 数据管理器
# ==========================================

class DynamicClusterDataset:
    """
    动态聚类数据集
    - 支持标签动态更新
    - 支持噪声点标记和过滤
    - 维护簇中心嵌入
    """
    
    def __init__(self, data_dir: str, seq_len: int = 150):
        self.seq_len = seq_len
        self.base_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.idx_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        
        # 核心数据结构
        self.reads = []           # 所有reads的序列
        self.read_ids = []        # read标识
        self.labels = []          # 当前簇标签 (-1表示噪声)
        self.original_labels = [] # 初始标签（用于对比）
        self.confidence = []      # 每个read的置信度分数
        
        # 簇信息
        self.num_clusters = 0     # 当前有效簇数量
        self.cluster_centers = {} # 簇中心嵌入 {cluster_id: embedding}
        self.cluster_refs = {}    # 簇的代表序列（来自Clover）
        
        # 加载数据
        self._load_data(data_dir)
        
    def _load_data(self, data_dir: str):
        """加载FedDNA格式的数据"""
        read_path = os.path.join(data_dir, "read.txt")
        ref_path = os.path.join(data_dir, "ref.txt")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(data_dir, "reference.txt")
        
        print(f"📂 加载数据: {data_dir}")
        
        # 加载代表序列（簇中心）
        with open(ref_path, 'r') as f:
            refs = [line.strip() for line in f if line.strip()]
        
        # 加载reads
        with open(read_path, 'r') as f:
            content = f.read().strip()
        
        raw_clusters = content.split("===============================")
        
        read_idx = 0
        for cluster_id, cluster_block in enumerate(raw_clusters):
            if not cluster_block.strip() or cluster_id >= len(refs):
                continue
            
            reads_in_cluster = [r.strip() for r in cluster_block.strip().split('\n') if r.strip()]
            
            # 保存簇的代表序列
            self.cluster_refs[cluster_id] = refs[cluster_id]
            
            for read_seq in reads_in_cluster:
                self.reads.append(read_seq)
                self.read_ids.append(f"read_{read_idx}")
                self.labels.append(cluster_id)
                self.original_labels.append(cluster_id)
                self.confidence.append(1.0)  # 初始置信度为1
                read_idx += 1
        
        self.num_clusters = len(self.cluster_refs)
        
        # 转换为numpy数组便于操作
        self.labels = np.array(self.labels)
        self.original_labels = np.array(self.original_labels)
        self.confidence = np.array(self.confidence)
        
        print(f"✅ 加载完成:")
        print(f"   - Reads: {len(self.reads)}")
        print(f"   - 初始簇数量 K: {self.num_clusters}")
        print(f"   - Reads/簇: {len(self.reads) / self.num_clusters:.1f}")
    
    def one_hot_encode(self, seq: str) -> np.ndarray:
        """序列转one-hot编码"""
        arr = np.zeros((self.seq_len, 4), dtype=np.float32)
        for i, char in enumerate(seq[:self.seq_len]):
            if char in self.base_mapping:
                arr[i, self.base_mapping[char]] = 1.0
        return arr
    
    def get_cluster_reads(self, cluster_id: int, exclude_noise: bool = True) -> List[int]:
        """获取某个簇的所有read索引"""
        if exclude_noise:
            return [i for i, l in enumerate(self.labels) if l == cluster_id and l != -1]
        return [i for i, l in enumerate(self.labels) if l == cluster_id]
    
    def get_valid_reads(self) -> List[int]:
        """获取所有非噪声的read索引"""
        return [i for i, l in enumerate(self.labels) if l != -1]
    
    def get_active_clusters(self) -> List[int]:
        """获取当前有效的簇ID列表"""
        active = set(self.labels[self.labels != -1])
        return sorted(list(active))
    
    def update_label(self, read_idx: int, new_label: int):
        """更新单个read的标签"""
        self.labels[read_idx] = new_label
    
    def mark_as_noise(self, read_idx: int):
        """将read标记为噪声"""
        self.labels[read_idx] = -1
    
    def update_confidence(self, read_idx: int, conf: float):
        """更新置信度"""
        self.confidence[read_idx] = conf
    
    def get_statistics(self) -> Dict:
        """获取当前数据集统计信息"""
        active_clusters = self.get_active_clusters()
        noise_count = np.sum(self.labels == -1)
        
        cluster_sizes = []
        for cid in active_clusters:
            size = np.sum(self.labels == cid)
            cluster_sizes.append(size)
        
        return {
            'total_reads': len(self.reads),
            'active_clusters': len(active_clusters),
            'noise_count': noise_count,
            'noise_ratio': noise_count / len(self.reads),
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': np.min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': np.max(cluster_sizes) if cluster_sizes else 0,
        }

# ==========================================
# 编码器
# ==========================================

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

class Conv2dUpsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
    
    def forward(self, x):
        # x: [B, L, 4]
        x = x.unsqueeze(1)  # [B, 1, L, 4]
        x = self.conv(x)    # [B, C, L, 4]
        B, C, L, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, L, C, D]
        x = x.view(B, L, C * D)  # [B, L, C*D]
        return x

class DNAEncoder(nn.Module):
    """DNA序列编码器"""
    
    def __init__(self, hidden_dim: int = 128, seq_len: int = 150):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.upsampling = Conv2dUpsampling(1, hidden_dim // 4, dropout_p=0.1)
        
        # 特征维度适配
        self.feature_proj = nn.Linear(hidden_dim // 4 * 4, hidden_dim)
        
        # 使用ConmambaBlock或LSTM
        if ConmambaBlock is not None:
            self.encoder = ConmambaBlock(
                dim=hidden_dim, ff_mult=4, conv_expansion_factor=2,
                conv_kernel_size=31, attn_dropout=0.1, ff_dropout=0.1, conv_dropout=0.1
            )
        else:
            self.encoder = nn.LSTM(
                hidden_dim, hidden_dim // 2, num_layers=2,
                batch_first=True, bidirectional=True, dropout=0.1
            )
        
        # 投影到嵌入空间
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        """
        x: [B, L, 4] - one-hot编码的序列
        返回: 
            - sequence_features: [B, L, hidden_dim] - 序列级特征
            - embedding: [B, hidden_dim] - 全局嵌入
        """
        # 上采样
        x = self.upsampling(x)  # [B, L, C*4]
        x = self.feature_proj(x)  # [B, L, hidden_dim]
        
        # 编码
        if isinstance(self.encoder, nn.LSTM):
            x, _ = self.encoder(x)  # [B, L, hidden_dim]
        else:
            x = self.encoder(x)  # [B, L, hidden_dim]
        
        sequence_features = x
        
        # 全局嵌入 (平均池化)
        embedding = torch.mean(x, dim=1)  # [B, hidden_dim]
        embedding = self.embed_proj(embedding)  # [B, hidden_dim]
        
        return sequence_features, embedding

# ==========================================
# 证据解码器
# ==========================================

class EvidenceDecoder(nn.Module):
    """
    证据解码器
    - 为每个位置生成4个碱基的证据
    - 计算证据强度（置信度）
    """
    
    def __init__(self, hidden_dim: int = 128, seq_len: int = 150):
        super().__init__()
        
        self.rnn = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=2,
            batch_first=True, dropout=0.1
        )
        
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
            nn.Softplus()  # 保证证据非负
        )
        
        self.min_evidence = 0.1
        
    def forward(self, sequence_features):
        """
        sequence_features: [B, L, hidden_dim]
        返回:
            - evidence: [B, L, 4] - 每个位置的证据
            - strength: [B] - 每个样本的证据强度
        """
        x, _ = self.rnn(sequence_features)
        evidence = self.evidence_head(x) + self.min_evidence  # [B, L, 4]
        
        # 计算证据强度 (总证据量)
        strength = torch.sum(evidence, dim=(1, 2))  # [B]
        
        # 归一化强度到[0, 1]范围
        strength = torch.sigmoid(strength / 1000 - 3)  # 调整阈值
        
        return evidence, strength

# ==========================================
# 对比学习模块
# ==========================================

class ContrastiveLearning(nn.Module):
    """
    监督对比学习
    - 同一簇的reads拉近
    - 不同簇的reads推远
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, D] - L2归一化的嵌入
        labels: [B] - 簇标签
        """
        # L2归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature  # [B, B]
        
        # 创建标签掩码
        labels = labels.view(-1, 1)
        mask_positive = (labels == labels.T).float()  # 同簇为1
        mask_negative = 1 - mask_positive
        
        # 移除对角线
        eye = torch.eye(embeddings.shape[0], device=embeddings.device)
        mask_positive = mask_positive - eye
        
        # InfoNCE损失
        exp_sim = torch.exp(sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0])
        
        # 分子：正样本对的相似度
        pos_sim = (exp_sim * mask_positive).sum(dim=1)
        
        # 分母：所有样本对（除了自己）
        all_sim = (exp_sim * (1 - eye)).sum(dim=1)
        
        # 避免除零
        loss = -safe_log(safe_div(pos_sim, all_sim))
        
        # 只计算有正样本的
        valid_mask = mask_positive.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            return loss[valid_mask].mean()
        return torch.tensor(0.0, device=embeddings.device)

# ==========================================
# 证据融合
# ==========================================

def evidence_fusion(evidences: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    加权证据融合
    
    evidences: [N, L, 4] - N条reads的证据
    weights: [N] - 每条read的权重（基于强度）
    
    返回: [L, 4] - 融合后的证据
    """
    # 归一化权重
    weights = F.softmax(weights, dim=0)  # [N]
    
    # 加权融合
    weights = weights.view(-1, 1, 1)  # [N, 1, 1]
    fused = torch.sum(evidences * weights, dim=0)  # [L, 4]
    
    return fused

# ==========================================
# 主模型
# ==========================================

class EvidenceDrivenClusteringModel(nn.Module):
    """
    证据驱动聚类模型
    
    功能:
    1. 编码reads得到嵌入
    2. 生成证据和强度
    3. 对比学习优化特征空间
    4. 融合证据进行预测
    """
    
    def __init__(self, hidden_dim: int = 128, seq_len: int = 150):
        super().__init__()
        
        self.encoder = DNAEncoder(hidden_dim, seq_len)
        self.decoder = EvidenceDecoder(hidden_dim, seq_len)
        self.contrastive = ContrastiveLearning(temperature=0.1)
        
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
    def forward(self, reads: torch.Tensor):
        """
        reads: [B, L, 4]
        返回:
            - embeddings: [B, hidden_dim]
            - evidence: [B, L, 4]
            - strength: [B]
        """
        sequence_features, embeddings = self.encoder(reads)
        evidence, strength = self.decoder(sequence_features)
        
        return embeddings, evidence, strength
    
    def compute_losses(self, reads: torch.Tensor, labels: torch.Tensor, 
                       ref_evidence: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        计算所有损失
        
        reads: [B, L, 4]
        labels: [B] - 簇标签
        ref_evidence: [L, 4] - 参考证据（融合后的）
        """
        embeddings, evidence, strength = self.forward(reads)
        
        losses = {}
        
        # 1. 对比学习损失
        losses['contrastive'] = self.contrastive(embeddings, labels)
        
        # 2. 证据一致性损失（同一簇的evidence应该相似）
        unique_labels = torch.unique(labels[labels >= 0])
        consistency_loss = torch.tensor(0.0, device=reads.device)
        count = 0
        
        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 1:
                cluster_evidence = evidence[mask]  # [n, L, 4]
                mean_evidence = cluster_evidence.mean(dim=0, keepdim=True)  # [1, L, 4]
                diff = cluster_evidence - mean_evidence
                consistency_loss = consistency_loss + torch.mean(diff ** 2)
                count += 1
        
        if count > 0:
            losses['consistency'] = consistency_loss / count
        else:
            losses['consistency'] = torch.tensor(0.0, device=reads.device)
        
        # 3. 如果有参考证据，计算重建损失
        if ref_evidence is not None:
            # KL散度
            alpha_pred = evidence + 1.0
            alpha_ref = ref_evidence.unsqueeze(0) + 1.0
            
            S_pred = alpha_pred.sum(dim=-1, keepdim=True)
            S_ref = alpha_ref.sum(dim=-1, keepdim=True)
            
            prob_pred = alpha_pred / S_pred
            prob_ref = alpha_ref / S_ref
            
            kl = torch.sum(prob_pred * (safe_log(prob_pred) - safe_log(prob_ref)), dim=-1)
            losses['reconstruction'] = kl.mean()
        
        # 4. 总损失
        losses['total'] = (
            losses['contrastive'] + 
            0.5 * losses['consistency'] + 
            losses.get('reconstruction', torch.tensor(0.0, device=reads.device))
        )
        
        return losses, embeddings, evidence, strength

# ==========================================
# 训练器
# ==========================================

class EvidenceDrivenTrainer:
    """
    证据驱动训练器
    
    训练流程:
    1. Mini-batch训练（对比学习 + 证据生成）
    2. 困难样本检测（基于证据强度）
    3. 标签修正（重分配或标记噪声）
    """
    
    def __init__(self, 
                 model: EvidenceDrivenClusteringModel,
                 dataset: DynamicClusterDataset,
                 device: torch.device,
                 lr: float = 1e-4,
                 confidence_threshold: float = 0.3,
                 noise_distance_threshold: float = 2.0):
        
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        
        # 阈值
        self.confidence_threshold = confidence_threshold  # 低于此值为困难样本
        self.noise_distance_threshold = noise_distance_threshold  # 高于此值为噪声
        
        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        
        # 簇中心嵌入
        self.cluster_centers = {}
        
        # 历史记录
        self.history = {
            'epoch': [],
            'total_loss': [],
            'contrastive_loss': [],
            'consistency_loss': [],
            'refinement_count': [],
            'noise_count': [],
            'active_clusters': []
        }
    
    def compute_cluster_centers(self):
        """计算所有簇的中心嵌入"""
        self.model.eval()
        
        cluster_embeddings = defaultdict(list)
        
        with torch.no_grad():
            for cid in self.dataset.get_active_clusters():
                read_indices = self.dataset.get_cluster_reads(cid)
                
                if len(read_indices) == 0:
                    continue
                
                # 分批处理
                batch_size = 32
                embeddings = []
                
                for i in range(0, len(read_indices), batch_size):
                    batch_indices = read_indices[i:i+batch_size]
                    reads = [self.dataset.reads[idx] for idx in batch_indices]
                    reads_encoded = torch.tensor(
                        np.array([self.dataset.one_hot_encode(r) for r in reads])
                    ).to(self.device)
                    
                    emb, _, _ = self.model(reads_encoded)
                    embeddings.append(emb.cpu())
                
                all_emb = torch.cat(embeddings, dim=0)
                self.cluster_centers[cid] = all_emb.mean(dim=0)
        
        print(f"   ✅ 更新了 {len(self.cluster_centers)} 个簇中心")
    
    def train_epoch(self, batch_size: int = 16, reads_per_cluster: int = 4) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        # 获取有效簇
        active_clusters = self.dataset.get_active_clusters()
        
        if len(active_clusters) < 2:
            print("⚠️ 有效簇数量不足，跳过训练")
            return epoch_losses
        
        # 随机采样batch
        num_iterations = len(self.dataset.get_valid_reads()) // (batch_size * reads_per_cluster)
        num_iterations = max(num_iterations, 10)
        
        for _ in range(num_iterations):
            # 随机选择簇
            selected_clusters = random.sample(
                active_clusters, 
                min(batch_size, len(active_clusters))
            )
            
            # 从每个簇中采样reads
            batch_reads = []
            batch_labels = []
            
            for cid in selected_clusters:
                cluster_reads = self.dataset.get_cluster_reads(cid)
                if len(cluster_reads) < reads_per_cluster:
                    selected_reads = cluster_reads
                else:
                    selected_reads = random.sample(cluster_reads, reads_per_cluster)
                
                for idx in selected_reads:
                    batch_reads.append(self.dataset.one_hot_encode(self.dataset.reads[idx]))
                    batch_labels.append(cid)
            
            if len(batch_reads) < 4:
                continue
            
            # 转换为tensor
            reads_tensor = torch.tensor(np.array(batch_reads)).to(self.device)
            labels_tensor = torch.tensor(batch_labels).to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            losses, embeddings, evidence, strength = self.model.compute_losses(
                reads_tensor, labels_tensor
            )
            
            # 反向传播
            loss = losses['total']
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            num_batches += 1
            
            # 更新置信度（用于后续refinement）
            for i, idx in enumerate([self.dataset.get_cluster_reads(cid) 
                                     for cid in selected_clusters for _ in range(reads_per_cluster)]):
                if i < len(strength):
                    # 这里简化处理，实际应该跟踪具体的read索引
                    pass
        
        # 平均损失
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        return dict(epoch_losses)
    
    def refine_labels(self) -> Tuple[int, int]:
        """
        困难样本修正
        
        返回: (修正数量, 新噪声数量)
        """
        self.model.eval()
        
        # 首先更新簇中心
        self.compute_cluster_centers()
        
        if len(self.cluster_centers) == 0:
            return 0, 0
        
        refinement_count = 0
        new_noise_count = 0
        
        # 处理所有非噪声的reads
        valid_indices = self.dataset.get_valid_reads()
        
        with torch.no_grad():
            # 分批处理
            batch_size = 64
            
            for start in range(0, len(valid_indices), batch_size):
                batch_indices = valid_indices[start:start+batch_size]
                
                reads = [self.dataset.reads[idx] for idx in batch_indices]
                reads_encoded = torch.tensor(
                    np.array([self.dataset.one_hot_encode(r) for r in reads])
                ).to(self.device)
                
                # 获取嵌入和强度
                embeddings, _, strength = self.model(reads_encoded)
                embeddings = embeddings.cpu()
                strength = strength.cpu().numpy()
                
                # 处理每个样本
                for i, (idx, emb, conf) in enumerate(zip(batch_indices, embeddings, strength)):
                    current_label = self.dataset.labels[idx]
                    
                    # 更新置信度
                    self.dataset.update_confidence(idx, conf)
                    
                    # 只处理低置信度样本
                    if conf >= self.confidence_threshold:
                        continue
                    
                    # 计算到所有簇中心的距离
                    distances = {}
                    for cid, center in self.cluster_centers.items():
                        dist = torch.norm(emb - center).item()
                        distances[cid] = dist
                    
                    if not distances:
                        continue
                    
                    # 找最近的簇
                    nearest_cluster = min(distances, key=distances.get)
                    min_distance = distances[nearest_cluster]
                    
                    # 判断是否需要修正
                    if min_distance > self.noise_distance_threshold:
                        # 太远，标记为噪声
                        self.dataset.mark_as_noise(idx)
                        new_noise_count += 1
                    elif nearest_cluster != current_label:
                        # 重分配到更近的簇
                        self.dataset.update_label(idx, nearest_cluster)
                        refinement_count += 1
        
        return refinement_count, new_noise_count
    
    def train(self, 
              max_epochs: int = 30,
              refinement_interval: int = 3,
              convergence_threshold: float = 0.01) -> Dict:
        """
        完整训练流程
        """
        print("="*60)
        print("🧬 证据驱动DNA聚类优化")
        print("="*60)
        
        # 初始统计
        stats = self.dataset.get_statistics()
        print(f"\n📊 初始状态:")
        print(f"   - 总Reads: {stats['total_reads']}")
        print(f"   - 簇数量 K: {stats['active_clusters']}")
        print(f"   - 置信度阈值: {self.confidence_threshold}")
        print(f"   - 噪声距离阈值: {self.noise_distance_threshold}")
        
        for epoch in range(max_epochs):
            print(f"\n{'='*60}")
            print(f"📍 Epoch {epoch + 1}/{max_epochs}")
            print(f"{'='*60}")
            
            # 训练
            losses = self.train_epoch()
            
            print(f"\n📈 训练损失:")
            print(f"   - Total: {losses.get('total', 0):.4f}")
            print(f"   - Contrastive: {losses.get('contrastive', 0):.4f}")
            print(f"   - Consistency: {losses.get('consistency', 0):.4f}")
            
            # 记录历史
            self.history['epoch'].append(epoch + 1)
            self.history['total_loss'].append(losses.get('total', 0))
            self.history['contrastive_loss'].append(losses.get('contrastive', 0))
            self.history['consistency_loss'].append(losses.get('consistency', 0))
            
            # 标签修正
            if (epoch + 1) % refinement_interval == 0:
                print(f"\n🔧 标签修正...")
                refinement_count, new_noise = self.refine_labels()
                
                stats = self.dataset.get_statistics()
                
                print(f"   - 重分配: {refinement_count} reads")
                print(f"   - 新噪声: {new_noise} reads")
                print(f"   - 当前噪声率: {stats['noise_ratio']*100:.2f}%")
                print(f"   - 有效簇数: {stats['active_clusters']}")
                
                self.history['refinement_count'].append(refinement_count)
                self.history['noise_count'].append(stats['noise_count'])
                self.history['active_clusters'].append(stats['active_clusters'])
                
                # 收敛检查
                total_valid = len(self.dataset.get_valid_reads())
                refinement_ratio = refinement_count / max(total_valid, 1)
                
                if refinement_ratio < convergence_threshold:
                    print(f"\n✅ 收敛! 修正比例 {refinement_ratio*100:.2f}% < {convergence_threshold*100}%")
                    break
            
            self.scheduler.step()
        
        # 最终统计
        print(f"\n{'='*60}")
        print("🎉 训练完成!")
        print(f"{'='*60}")
        
        final_stats = self.dataset.get_statistics()
        print(f"\n📊 最终状态:")
        print(f"   - 有效Reads: {final_stats['total_reads'] - final_stats['noise_count']}")
        print(f"   - 噪声Reads: {final_stats['noise_count']} ({final_stats['noise_ratio']*100:.1f}%)")
        print(f"   - 最终簇数: {final_stats['active_clusters']}")
        print(f"   - 平均簇大小: {final_stats['avg_cluster_size']:.1f}")
        
        return self.history
    
    def get_final_clustering(self) -> Dict[int, List[str]]:
        """获取最终聚类结果"""
        result = defaultdict(list)
        
        for i, (read, label) in enumerate(zip(self.dataset.reads, self.dataset.labels)):
            if label != -1:
                result[label].append(read)
        
        return dict(result)

# ==========================================
# 可视化
# ==========================================

def plot_training_history(history: Dict, output_path: str = "training_history.png"):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Evidence-Driven Clustering Training', fontsize=14, fontweight='bold')
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(history['epoch'], history['total_loss'], 'b-', label='Total', linewidth=2)
    ax1.plot(history['epoch'], history['contrastive_loss'], 'r--', label='Contrastive', linewidth=2)
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 修正数量
    ax2 = axes[0, 1]
    if history['refinement_count']:
        refinement_epochs = [e for i, e in enumerate(history['epoch']) if (i+1) % 3 == 0][:len(history['refinement_count'])]
        ax2.bar(refinement_epochs, history['refinement_count'], alpha=0.7, color='orange')
    ax2.set_title('Label Refinements per Interval')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # 3. 噪声数量
    ax3 = axes[1, 0]
    if history['noise_count']:
        noise_epochs = refinement_epochs[:len(history['noise_count'])]
        ax3.plot(noise_epochs, history['noise_count'], 'r-o', linewidth=2)
    ax3.set_title('Noise Count')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)
    
    # 4. 有效簇数
    ax4 = axes[1, 1]
    if history['active_clusters']:
        cluster_epochs = refinement_epochs[:len(history['active_clusters'])]
        ax4.plot(cluster_epochs, history['active_clusters'], 'g-o', linewidth=2)
    ax4.set_title('Active Clusters (K)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('K')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 图表已保存: {output_path}")

# ==========================================
# 主函数
# ==========================================

def main():
    print("="*60)
    print("🧬 证据驱动DNA聚类优化模型")
    print("="*60)
    
    # 配置
    DATA_DIR = "CC/Step0/Experiments/20251216_145746_Improved_Data_Test/03_FedDNA_In"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 设备: {device}")
    
    # 随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 加载数据
    dataset = DynamicClusterDataset(DATA_DIR, seq_len=150)
    
    # 创建模型
    model = EvidenceDrivenClusteringModel(
        hidden_dim=128,
        seq_len=150
    )
    
    # 创建训练器
    trainer = EvidenceDrivenTrainer(
        model=model,
        dataset=dataset,
        device=device,
        lr=1e-4,
        confidence_threshold=0.3,   # 低于30%置信度为困难样本
        noise_distance_threshold=2.0  # 距离阈值
    )
    
    # 训练
    history = trainer.train(
        max_epochs=30,
        refinement_interval=3,
        convergence_threshold=0.01
    )
    
    # 保存结果
    print("\n💾 保存结果...")
    
    # 保存聚类结果
    final_clusters = trainer.get_final_clustering()
    with open("final_clustering.txt", 'w') as f:
        for cid, reads in final_clusters.items():
            f.write(f"=== Cluster {cid} ({len(reads)} reads) ===\n")
            for read in reads[:5]:  # 只保存前5条作为示例
                f.write(f"{read}\n")
            f.write("\n")
    print(f"✅ 聚类结果: final_clustering.txt")
    
    # 保存标签
    with open("final_labels.txt", 'w') as f:
        f.write("read_id\toriginal_label\tfinal_label\tconfidence\n")
        for i in range(len(dataset.reads)):
            f.write(f"{dataset.read_ids[i]}\t{dataset.original_labels[i]}\t"
                    f"{dataset.labels[i]}\t{dataset.confidence[i]:.4f}\n")
    print(f"✅ 标签文件: final_labels.txt")
    
    # 绘制训练曲线
    plot_training_history(history)
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"evidence_driven_model_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'final_stats': dataset.get_statistics()
    }, model_path)
    print(f"✅ 模型: {model_path}")
    
    return model, dataset, history

if __name__ == "__main__":
    model, dataset, history = main()
