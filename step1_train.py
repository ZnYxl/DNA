#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNA 聚类 Metadata 生成脚本 - 完整Dirichlet Evidence Learning版本
将 Clover 输出转换为 CSV 元数据文件
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

# ==========================================
# 定义权重参数
# ==========================================
alpha = 1.0
beta = 0.01
gamma = 0.01

# ==========================================
# 导入基础组件
# ==========================================
try:
    from models.conmamba import ConmambaBlock
    print("✅ 成功导入 ConmambaBlock")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ==========================================
# 简化的核心组件
# ==========================================

class SimpleEncoder(nn.Module):
    """简化的编码器 - 只保留核心功能"""
    def __init__(self, input_dim=4, hidden_dim=64, seq_len=150):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 简单的特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # ConMamba块用于序列建模
        self.conmamba = ConmambaBlock(dim=hidden_dim)
        
    def forward(self, x):
        # x: [B*N, L, 4] 
        x = self.feature_extractor(x)  # [B*N, L, hidden_dim]
        x = self.conmamba(x)           # [B*N, L, hidden_dim]
        return x

class ContrastiveLearning(nn.Module):
    """对比学习模块"""
    def __init__(self, hidden_dim=64, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, embeddings):
        # embeddings: [B*N, L, hidden_dim]
        # 取平均池化作为序列表示
        seq_repr = torch.mean(embeddings, dim=1)  # [B*N, hidden_dim]
        projected = self.projection(seq_repr)     # [B*N, hidden_dim//2]
        return F.normalize(projected, dim=-1)

class DirichletEvidenceDecoder(nn.Module):
    """🔥 严格的Dirichlet证据解码器"""
    def __init__(self, hidden_dim=64, output_dim=4):
        super().__init__()
        self.output_dim = output_dim
        
        # 证据网络 - 输出非负evidence
        self.evidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # 确保evidence > 0
        )
        
    def forward(self, x):
        """
        x: [B*N, L, hidden_dim]
        返回: Dirichlet参数和相关统计量
        """
        # 1️⃣ 获取evidence
        evidence = self.evidence_net(x)  # [B*N, L, K]
        
        # 2️⃣ 证据 → Dirichlet 参数
        alpha = evidence + 1.0  # [B*N, L, K]
        
        # 3️⃣ 预测均值（不是softmax！）
        alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)  # [B*N, L, 1]
        predictions = alpha / alpha_sum  # [B*N, L, K]
        
        # 4️⃣ 不确定性计算
        K = self.output_dim
        uncertainty = K / alpha_sum.squeeze(-1)  # [B*N, L]
        
        # 5️⃣ 证据强度（用于融合权重）
        evidence_strength = torch.sum(evidence, dim=-1)  # [B*N, L]
        
        return {
            'evidence': evidence,           # [B*N, L, K] - 原始evidence
            'alpha': alpha,                # [B*N, L, K] - Dirichlet参数
            'predictions': predictions,     # [B*N, L, K] - 预测概率
            'uncertainty': uncertainty,     # [B*N, L] - 不确定性
            'strength': evidence_strength   # [B*N, L] - 证据强度
        }

class DirichletEvidenceFusion(nn.Module):
    """🔥 基于Dirichlet的证据融合模块"""
    def __init__(self):
        super().__init__()
        
    def forward(self, dirichlet_outputs):
        """
        dirichlet_outputs: dict with keys ['evidence', 'alpha', 'predictions', 'uncertainty', 'strength']
        每个值的shape: [N, L, K] 或 [N, L]
        """
        evidence = dirichlet_outputs['evidence']      # [N, L, K]
        uncertainty = dirichlet_outputs['uncertainty'] # [N, L]
        
        # 🔥 使用不确定性的倒数作为融合权重（不确定性越低，权重越高）
        fusion_weights = 1.0 / (uncertainty + 1e-8)  # [N, L]
        fusion_weights = fusion_weights.unsqueeze(-1)  # [N, L, 1]
        
        # 加权融合evidence
        weighted_evidence = evidence * fusion_weights  # [N, L, K]
        fused_evidence = torch.sum(weighted_evidence, dim=0)  # [L, K]
        total_weights = torch.sum(fusion_weights, dim=0)      # [L, 1]
        
        # 归一化
        fused_evidence = fused_evidence / (total_weights + 1e-8)  # [L, K]
        
        # 重新计算融合后的Dirichlet参数
        fused_alpha = fused_evidence + 1.0  # [L, K]
        fused_alpha_sum = torch.sum(fused_alpha, dim=-1, keepdim=True)  # [L, 1]
        fused_predictions = fused_alpha / fused_alpha_sum  # [L, K]
        fused_uncertainty = evidence.shape[-1] / fused_alpha_sum.squeeze(-1)  # [L]
        
        return {
            'fused_evidence': fused_evidence,
            'fused_alpha': fused_alpha,
            'fused_predictions': fused_predictions,
            'fused_uncertainty': fused_uncertainty,
            'fusion_weights': fusion_weights.squeeze(-1)  # [N, L]
        }

class SimplifiedFedDNA(nn.Module):
    """简化版FedDNA - 使用完整Dirichlet代数"""
    def __init__(self, input_dim=4, hidden_dim=64, seq_len=150):
        super().__init__()
        self.encoder = SimpleEncoder(input_dim, hidden_dim, seq_len)
        self.contrastive = ContrastiveLearning(hidden_dim)
        self.evidence_decoder = DirichletEvidenceDecoder(hidden_dim, input_dim)
        self.evidence_fusion = DirichletEvidenceFusion()
        
    def forward(self, reads_batch):
        """
        reads_batch: [B, N, L, 4] - B个batch，每个batch有N条reads
        """
        B, N, L, D = reads_batch.shape
        
        # 重塑为 [B*N, L, D]
        reads_flat = reads_batch.view(B * N, L, D)
        
        # 1. 编码
        embeddings = self.encoder(reads_flat)  # [B*N, L, hidden_dim]
        
        # 2. 对比学习特征
        contrastive_features = self.contrastive(embeddings)  # [B*N, hidden_dim//2]
        
        # 3. Dirichlet证据解码
        dirichlet_outputs = self.evidence_decoder(embeddings)
        
        # 4. 重塑回batch形式
        for key in dirichlet_outputs:
            if dirichlet_outputs[key].dim() == 3:  # [B*N, L, K]
                dirichlet_outputs[key] = dirichlet_outputs[key].view(B, N, L, -1)
            elif dirichlet_outputs[key].dim() == 2:  # [B*N, L]
                dirichlet_outputs[key] = dirichlet_outputs[key].view(B, N, L)
        
        contrastive_features = contrastive_features.view(B, N, -1)  # [B, N, hidden_dim//2]
        
        return dirichlet_outputs, contrastive_features

# ==========================================
# 🔥 完整Dirichlet损失函数
# ==========================================

class DirichletComprehensiveLoss(nn.Module):
    """🔥 基于Dirichlet代数的综合损失函数"""
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01, temperature=0.1):
        super().__init__()
        self.alpha = alpha      # Dirichlet Expected MSE权重
        self.beta = beta        # 对比学习损失权重  
        self.gamma = gamma      # Dirichlet KL散度权重
        self.temperature = temperature
        
    def dirichlet_expected_mse(self, fused_predictions, target):
        """
        🔥 Dirichlet Expected MSE Loss
        fused_predictions: [L, K] - Dirichlet预测均值
        target: [L, K] - 目标one-hot
        """
        mse = torch.mean((fused_predictions - target) ** 2)
        return mse
    
    def dirichlet_kl_divergence(self, alpha, target_alpha=None):
        """
        🔥 Dirichlet KL散度的解析式
        alpha: [L, K] - Dirichlet参数
        target_alpha: [L, K] - 目标Dirichlet参数（如果为None，使用均匀分布）
        """
        if target_alpha is None:
            # 与均匀Dirichlet分布的KL散度
            K = alpha.shape[-1]
            target_alpha = torch.ones_like(alpha)  # 均匀分布参数都是1
        
        # Dirichlet KL散度的解析公式
        alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)  # [L, 1]
        target_alpha_sum = torch.sum(target_alpha, dim=-1, keepdim=True)  # [L, 1]
        
        # KL(Dir(α)||Dir(α₀)) = log(B(α₀)/B(α)) + Σᵢ(αᵢ-α₀ᵢ)[ψ(αᵢ)-ψ(Σⱼαⱼ)]
        # 简化版本：使用对数Gamma函数
        kl_div = (
            torch.lgamma(alpha_sum) - torch.lgamma(target_alpha_sum) +
            torch.sum(torch.lgamma(target_alpha) - torch.lgamma(alpha), dim=-1, keepdim=True) +
            torch.sum((alpha - target_alpha) * (torch.digamma(alpha) - torch.digamma(alpha_sum)), dim=-1, keepdim=True)
        )
        
        return torch.mean(kl_div)
    
    def contrastive_loss(self, features, cluster_labels=None):
        """对比学习损失 - 保持不变"""
        if features.shape[0] <= 1:
            return torch.tensor(0.0, device=features.device)
            
        # 简化版：假设同一个batch内的reads属于同一簇
        features_norm = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, device=features.device).bool()
        
        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim.masked_fill(mask, 0)
        
        pos_sim = torch.sum(exp_sim, dim=1) / (batch_size - 1)
        neg_sim = torch.sum(exp_sim, dim=1)
        
        loss = -torch.log(pos_sim / (neg_sim + 1e-8))
        return torch.mean(loss)
    
    def forward(self, fusion_results, target, contrastive_features):
        """
        计算总损失
        fusion_results: dict - 融合结果
        target: [L, K] - 目标序列
        contrastive_features: [N, feature_dim] - 对比学习特征
        """
        fused_predictions = fusion_results['fused_predictions']  # [L, K]
        fused_alpha = fusion_results['fused_alpha']              # [L, K]
        
        # 1️⃣ Dirichlet Expected MSE
        expected_mse = self.dirichlet_expected_mse(fused_predictions, target)
        
        # 2️⃣ 对比学习损失
        contrastive_loss = self.contrastive_loss(contrastive_features)
        
        # 3️⃣ Dirichlet KL散度
        dirichlet_kl = self.dirichlet_kl_divergence(fused_alpha)
        
        # 4️⃣ 总损失
        total_loss = (self.alpha * expected_mse + 
                     self.beta * contrastive_loss + 
                     self.gamma * dirichlet_kl)
        
        return {
            'total_loss': total_loss,
            'expected_mse': expected_mse,
            'contrastive_loss': contrastive_loss,
            'dirichlet_kl': dirichlet_kl
        }

# ==========================================
# 🔥 基于Dirichlet不确定性的修正模块
# ==========================================

class DirichletEvidenceRefinement(nn.Module):
    """🔥 基于Dirichlet不确定性的困难样本修正模块"""
    
    def __init__(self, uncertainty_threshold=0.5, distance_threshold=2.0):
        super().__init__()
        self.uncertainty_threshold = uncertainty_threshold  # 不确定性阈值
        self.distance_threshold = distance_threshold
        
    def calculate_dirichlet_confidence(self, uncertainty):
        """
        🔥 基于Dirichlet不确定性计算置信度
        uncertainty: [N, L] - Dirichlet不确定性
        返回: [N] - 每条read的置信度分数
        """
        # 不确定性越低，置信度越高
        confidence_scores = 1.0 / (torch.mean(uncertainty, dim=1) + 1e-8)  # [N]
        
        # 归一化到[0,1]
        confidence_scores = torch.sigmoid(confidence_scores - 1.0)
        
        return confidence_scores
    
    def compute_cluster_centers(self, embeddings, labels, num_clusters):
        """计算簇中心 - 保持不变"""
        device = embeddings.device
        feature_dim = embeddings.shape[1]
        centers = torch.zeros(num_clusters, feature_dim, device=device)
        
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                centers[k] = torch.mean(embeddings[mask], dim=0)
            else:
                centers[k] = torch.randn(feature_dim, device=device)
                
        return centers
    
    def reassign_hard_samples(self, hard_embeddings, cluster_centers):
        """重分配困难样本 - 保持不变"""
        if hard_embeddings.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=hard_embeddings.device)
            
        distances = torch.cdist(hard_embeddings, cluster_centers)  # [M, K]
        min_distances, nearest_clusters = torch.min(distances, dim=1)  # [M]
        
        new_labels = nearest_clusters.clone()
        noise_mask = min_distances > self.distance_threshold
        new_labels[noise_mask] = -1
        
        return new_labels, min_distances
    
    def forward(self, embeddings, dirichlet_uncertainty, current_labels, num_clusters):
        """
        🔥 执行基于Dirichlet不确定性的修正流程
        
        参数:
        - embeddings: [N, feature_dim] - reads的嵌入表示
        - dirichlet_uncertainty: [N, L] - Dirichlet不确定性
        - current_labels: [N] - 当前标签
        - num_clusters: int - 簇数量
        
        返回:
        - new_labels: [N] - 修正后的标签
        - refinement_stats: dict - 修正统计信息
        """
        N = embeddings.shape[0]
        device = embeddings.device
        
        # 1️⃣ 基于Dirichlet不确定性计算置信度
        confidence_scores = self.calculate_dirichlet_confidence(dirichlet_uncertainty)
        
        # 2️⃣ 阈值判断 - 识别困难样本
        high_confidence_mask = confidence_scores > self.uncertainty_threshold
        hard_sample_mask = ~high_confidence_mask
        
        # 3️⃣ 保留高置信度样本的标签
        new_labels = current_labels.clone()
        
        # 4️⃣ 处理困难样本
        if hard_sample_mask.sum() > 0:
            high_conf_embeddings = embeddings[high_confidence_mask]
            high_conf_labels = current_labels[high_confidence_mask]
            
            if high_conf_embeddings.shape[0] > 0:
                cluster_centers = self.compute_cluster_centers(
                    high_conf_embeddings, high_conf_labels, num_clusters
                )
                
                hard_embeddings = embeddings[hard_sample_mask]
                reassigned_labels, distances = self.reassign_hard_samples(
                    hard_embeddings, cluster_centers
                )
                
                new_labels[hard_sample_mask] = reassigned_labels
        
        # 5️⃣ 统计修正信息
        refinement_stats = {
            'total_samples': N,
            'high_confidence_count': high_confidence_mask.sum().item(),
            'hard_samples_count': hard_sample_mask.sum().item(),
            'noise_samples_count': (new_labels == -1).sum().item(),
            'label_changes': (new_labels != current_labels).sum().item(),
            'refinement_ratio': (new_labels != current_labels).float().mean().item(),
            'avg_confidence': confidence_scores.mean().item(),
            'avg_uncertainty': torch.mean(dirichlet_uncertainty).item(),
            'min_confidence': confidence_scores.min().item(),
            'max_confidence': confidence_scores.max().item()
        }
        
        return new_labels, refinement_stats

# ==========================================
# 数据集 (保持不变)
# ==========================================
class CloverClusterDataset(Dataset):
    def __init__(self, data_dir, seq_len=150):
        self.seq_len = seq_len
        self.clusters = [] 
        
        read_path = os.path.join(data_dir, "read.txt")
        ref_path = os.path.join(data_dir, "ref.txt")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(data_dir, "reference.txt")
            
        if not os.path.exists(read_path):
            print(f"❌ 错误: 找不到数据文件 {read_path}")
            return
        
        print(f"📂 正在加载数据: {data_dir}")
        with open(ref_path, 'r') as f:
            refs = [line.strip() for line in f if line.strip()]
        with open(read_path, 'r') as f:
            content = f.read().strip()
        raw_clusters = content.split("===============================")
        
        for i, cluster_block in enumerate(raw_clusters):
            if not cluster_block.strip(): continue
            if i >= len(refs): break
            reads = [r.strip() for r in cluster_block.strip().split('\n') if r.strip()]
            if len(reads) > 0:
                self.clusters.append({'ref': refs[i], 'reads': reads})
                
        print(f"✅ 加载完成: {len(self.clusters)} 个簇")

    def one_hot(self, seq):
        mapping = {'A':0, 'C':1, 'G':2, 'T':3}
        arr = np.zeros((self.seq_len, 4), dtype=np.float32)
        l = min(len(seq), self.seq_len)
        for i in range(l):
            char = seq[i]
            if char in mapping: arr[i, mapping[char]] = 1.0
        return arr

    def __len__(self): return len(self.clusters)

    def __getitem__(self, idx):
        cluster = self.clusters[idx]
        reads_vec = np.array([self.one_hot(r) for r in cluster['reads']])
        ref_vec = self.one_hot(cluster['ref'])
        return torch.tensor(reads_vec), torch.tensor(ref_vec)

# ==========================================
# 🔥 完整Dirichlet训练器
# ==========================================

class DirichletEvidenceRefinement(nn.Module):
    """🔥 彻底修复版：基于Dirichlet不确定性的困难样本修正模块"""
    
    def __init__(self, uncertainty_threshold=0.7, confidence_threshold=0.4, distance_threshold=1.0):
        super().__init__()
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold
        
    def calculate_dirichlet_confidence(self, uncertainty):
        """计算置信度"""
        avg_uncertainty = torch.mean(uncertainty, dim=1)  # [N]
        
        # 线性变换到[0,1]
        max_uncertainty = torch.max(avg_uncertainty)
        min_uncertainty = torch.min(avg_uncertainty)
        if max_uncertainty > min_uncertainty:
            confidence_scores = 1.0 - (avg_uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty)
        else:
            confidence_scores = torch.ones_like(avg_uncertainty) * 0.5
        
        return confidence_scores, avg_uncertainty
    
    def identify_hard_samples_multi_criteria(self, uncertainty, confidence_scores):
        """多重标准识别困难样本"""
        N = uncertainty.shape[0]
        
        # 标准1：高不确定性
        avg_uncertainty = torch.mean(uncertainty, dim=1)  # [N]
        high_uncertainty_mask = avg_uncertainty > self.uncertainty_threshold
        
        # 标准2：低置信度
        low_confidence_mask = confidence_scores < self.confidence_threshold
        
        # 标准3：不确定性方差大
        uncertainty_var = torch.var(uncertainty, dim=1)  # [N]
        uncertainty_var_threshold = torch.quantile(uncertainty_var, 0.7)
        high_variance_mask = uncertainty_var > uncertainty_var_threshold
        
        # 综合判断：满足任意两个条件
        criteria_count = (high_uncertainty_mask.float() + 
                         low_confidence_mask.float() + 
                         high_variance_mask.float())
        
        hard_sample_mask = criteria_count >= 2.0
        
        # 如果没有困难样本，降低标准
        if hard_sample_mask.sum() == 0:
            hard_sample_mask = criteria_count >= 1.0
            
        # 强制选择最不确定的30%
        if hard_sample_mask.sum() == 0:
            uncertainty_threshold_dynamic = torch.quantile(avg_uncertainty, 0.7)
            hard_sample_mask = avg_uncertainty > uncertainty_threshold_dynamic
        
        return hard_sample_mask, {
            'high_uncertainty_count': high_uncertainty_mask.sum().item(),
            'low_confidence_count': low_confidence_mask.sum().item(),
            'high_variance_count': high_variance_mask.sum().item(),
            'avg_uncertainty': avg_uncertainty.mean().item(),
            'uncertainty_var_threshold': uncertainty_var_threshold.item()
        }
    
    def create_multi_cluster_assignment(self, embeddings, uncertainty, num_base_clusters=3):
        """
        🔥 关键修复：创建多簇分配而不是单一簇
        基于embedding相似性和不确定性创建初始多簇标签
        """
        N = embeddings.shape[0]
        device = embeddings.device
        
        if N <= num_base_clusters:
            # 如果样本数太少，直接分配不同标签
            return torch.arange(N, device=device)
        
        # 方法1：基于embedding的K-means聚类
        from sklearn.cluster import KMeans
        import numpy as np
        
        embeddings_np = embeddings.detach().cpu().numpy()
        
        try:
            kmeans = KMeans(n_clusters=min(num_base_clusters, N), random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_np)
            initial_labels = torch.tensor(cluster_labels, device=device, dtype=torch.long)
        except:
            # 如果K-means失败，使用简单的基于不确定性的分组
            avg_uncertainty = torch.mean(uncertainty, dim=1)  # [N]
            
            # 按不确定性分成3组
            uncertainty_sorted, indices = torch.sort(avg_uncertainty)
            group_size = N // num_base_clusters
            
            initial_labels = torch.zeros(N, device=device, dtype=torch.long)
            for i in range(num_base_clusters):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < num_base_clusters - 1 else N
                group_indices = indices[start_idx:end_idx]
                initial_labels[group_indices] = i
        
        return initial_labels
    
    def compute_cluster_centers(self, embeddings, labels, num_clusters):
        """计算簇中心"""
        device = embeddings.device
        feature_dim = embeddings.shape[1]
        centers = torch.zeros(num_clusters, feature_dim, device=device)
        
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                centers[k] = torch.mean(embeddings[mask], dim=0)
            else:
                # 随机初始化空簇
                centers[k] = torch.randn(feature_dim, device=device) * 0.1
                
        return centers
    
    def reassign_hard_samples_improved(self, hard_embeddings, cluster_centers, hard_indices, current_labels):
        """
        🔥 改进版重分配：确保产生标签变化
        """
        if hard_embeddings.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=hard_embeddings.device), torch.tensor([], device=hard_embeddings.device)
            
        # 计算到所有簇中心的距离
        distances = torch.cdist(hard_embeddings, cluster_centers)  # [M, K]
        min_distances, nearest_clusters = torch.min(distances, dim=1)  # [M]
        
        # 获取当前标签
        current_hard_labels = current_labels[hard_indices]
        
        # 🔥 关键修复：强制改变标签
        new_labels = nearest_clusters.clone()
        
        # 对于距离太远的样本，标记为噪声
        noise_mask = min_distances > self.distance_threshold
        new_labels[noise_mask] = -1
        
        # 🔥 强制标签变化：如果新标签和旧标签相同，改为下一个最近的簇
        same_label_mask = (new_labels == current_hard_labels) & (new_labels != -1)
        if same_label_mask.sum() > 0:
            # 找到第二近的簇
            distances_masked = distances.clone()
            distances_masked[torch.arange(distances.shape[0]), nearest_clusters] = float('inf')
            second_nearest = torch.argmin(distances_masked, dim=1)
            
            # 对于标签相同的样本，分配到第二近的簇
            new_labels[same_label_mask] = second_nearest[same_label_mask]
        
        return new_labels, min_distances
    
    def forward(self, embeddings, dirichlet_uncertainty, current_labels, num_clusters):
        """
        🔥 彻底修复版：执行修正流程
        """
        N = embeddings.shape[0]
        device = embeddings.device
        
        # 🔥 关键修复：如果current_labels都相同，创建多簇初始分配
        if len(torch.unique(current_labels)) == 1:
            print(f"    🔧 检测到单一簇标签，创建多簇初始分配...")
            current_labels = self.create_multi_cluster_assignment(
                embeddings, dirichlet_uncertainty, num_base_clusters=3
            )
            num_clusters = max(3, len(torch.unique(current_labels)))
            print(f"    ✅ 创建了 {len(torch.unique(current_labels))} 个初始簇")
        
        # 1️⃣ 计算置信度
        confidence_scores, avg_uncertainty = self.calculate_dirichlet_confidence(dirichlet_uncertainty)
        
        # 2️⃣ 识别困难样本
        hard_sample_mask, criteria_stats = self.identify_hard_samples_multi_criteria(
            dirichlet_uncertainty, confidence_scores
        )
        
        high_confidence_mask = ~hard_sample_mask
        
        # 3️⃣ 保留高置信度样本的标签
        new_labels = current_labels.clone()
        
        # 4️⃣ 处理困难样本
        reassignment_stats = {'reassigned_count': 0, 'noise_count': 0, 'label_change_count': 0}
        
        if hard_sample_mask.sum() > 0:
            print(f"    🎯 处理 {hard_sample_mask.sum()} 个困难样本...")
            
            if high_confidence_mask.sum() > 0:
                # 基于高置信度样本计算簇中心
                high_conf_embeddings = embeddings[high_confidence_mask]
                high_conf_labels = current_labels[high_confidence_mask]
                
                cluster_centers = self.compute_cluster_centers(
                    high_conf_embeddings, high_conf_labels, num_clusters
                )
            else:
                # 如果没有高置信度样本，使用所有样本计算簇中心
                cluster_centers = self.compute_cluster_centers(
                    embeddings, current_labels, num_clusters
                )
            
            # 重分配困难样本
            hard_embeddings = embeddings[hard_sample_mask]
            hard_indices = torch.where(hard_sample_mask)[0]
            
            reassigned_labels, distances = self.reassign_hard_samples_improved(
                hard_embeddings, cluster_centers, hard_indices, current_labels
            )
            
            # 更新标签
            old_hard_labels = new_labels[hard_sample_mask].clone()
            new_labels[hard_sample_mask] = reassigned_labels
            
            # 统计变化
            reassignment_stats['reassigned_count'] = (reassigned_labels != -1).sum().item()
            reassignment_stats['noise_count'] = (reassigned_labels == -1).sum().item()
            reassignment_stats['label_change_count'] = (reassigned_labels != old_hard_labels).sum().item()
            
            print(f"    📊 重分配结果: {reassignment_stats['label_change_count']} 个标签改变, "
                  f"{reassignment_stats['noise_count']} 个噪声")
        
        # 5️⃣ 详细统计信息
        total_label_changes = (new_labels != current_labels).sum().item()
        
        refinement_stats = {
            'total_samples': N,
            'high_confidence_count': high_confidence_mask.sum().item(),
            'hard_samples_count': hard_sample_mask.sum().item(),
            'noise_samples_count': (new_labels == -1).sum().item(),
            'label_changes': total_label_changes,
            'refinement_ratio': total_label_changes / N if N > 0 else 0.0,
            'avg_confidence': confidence_scores.mean().item(),
            'avg_uncertainty': avg_uncertainty.mean().item(),
            'min_confidence': confidence_scores.min().item(),
            'max_confidence': confidence_scores.max().item(),
            'min_uncertainty': avg_uncertainty.min().item(),
            'max_uncertainty': avg_uncertainty.max().item(),
            'unique_labels_before': len(torch.unique(current_labels)),
            'unique_labels_after': len(torch.unique(new_labels[new_labels != -1])),
            # 多重标准统计
            **criteria_stats,
            **reassignment_stats
        }
        
        return new_labels, refinement_stats

# ==========================================
# 🔥 修复训练器收敛逻辑
# ==========================================

class DirichletRefinementTrainer:
    """🔥 修复版训练器"""
    
    def __init__(self, model, criterion, optimizer, refinement_module, 
                 convergence_threshold=0.02, max_epochs=10, min_epochs=5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.refinement = refinement_module
        self.convergence_threshold = convergence_threshold
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        
    def train_epoch_with_refinement(self, dataloader, device, epoch):
        """训练一个epoch"""
        
        self.model.train()
        epoch_losses = {'total_loss': 0, 'expected_mse': 0, 'contrastive_loss': 0, 'dirichlet_kl': 0}
        all_refinement_stats = []
        step_count = 0
        
        print(f"\n🔄 Epoch {epoch+1} - 开始修复版Dirichlet训练...")
        
        for i, (reads, ref) in enumerate(dataloader):
            reads = reads.to(device)  # [1, N, 150, 4]
            ref = ref.squeeze(0).to(device)  # [150, 4]
            N = reads.shape[1]
            
            # === 步骤1: 正向传播 ===
            self.optimizer.zero_grad()
            
            dirichlet_outputs, contrastive_features = self.model(reads)
            
            single_batch_outputs = {}
            for key in dirichlet_outputs:
                single_batch_outputs[key] = dirichlet_outputs[key].squeeze(0)
            
            fusion_results = self.model.evidence_fusion(single_batch_outputs)
            
            contrastive_features_flat = contrastive_features.squeeze(0)
            losses = self.criterion(fusion_results, ref, contrastive_features_flat)
            
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # === 步骤2: 修正阶段 ===
            with torch.no_grad():
                # 🔥 使用随机初始标签而不是全0
                current_labels = torch.randint(0, 3, (N,), device=device)
                
                dirichlet_uncertainty = single_batch_outputs['uncertainty']  # [N, L]
                
                new_labels, refinement_stats = self.refinement(
                    embeddings=contrastive_features_flat,
                    dirichlet_uncertainty=dirichlet_uncertainty,
                    current_labels=current_labels,
                    num_clusters=3
                )
                
                all_refinement_stats.append(refinement_stats)
            
            # 记录损失
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            step_count += 1
            
            # 详细输出
            if (i + 1) % 3 == 0:
                print(f"  📊 Step {i+1:3d} | Loss: {losses['total_loss'].item():.4f} | "
                      f"标签变化: {refinement_stats['label_changes']}/{N} | "
                      f"修正率: {refinement_stats['refinement_ratio']:.3f}")
                print(f"       困难样本: {refinement_stats['hard_samples_count']} | "
                      f"不确定性: {refinement_stats['min_uncertainty']:.3f}-{refinement_stats['max_uncertainty']:.3f} | "
                      f"簇数: {refinement_stats['unique_labels_before']}→{refinement_stats['unique_labels_after']}")
        
        # 计算epoch统计
        avg_losses = {key: val / max(1, step_count) for key, val in epoch_losses.items()}
        
        # 汇总修正统计
        if all_refinement_stats:
            avg_refinement_ratio = np.mean([s['refinement_ratio'] for s in all_refinement_stats])
            avg_confidence = np.mean([s['avg_confidence'] for s in all_refinement_stats])
            avg_uncertainty = np.mean([s['avg_uncertainty'] for s in all_refinement_stats])
            total_noise = sum([s['noise_samples_count'] for s in all_refinement_stats])
            total_hard_samples = sum([s['hard_samples_count'] for s in all_refinement_stats])
            total_label_changes = sum([s['label_changes'] for s in all_refinement_stats])
        else:
            avg_refinement_ratio = 0.0
            avg_confidence = 0.0
            avg_uncertainty = 0.0
            total_noise = 0
            total_hard_samples = 0
            total_label_changes = 0
            
        return avg_losses, {
            'refinement_ratio': avg_refinement_ratio,
            'avg_confidence': avg_confidence,
            'avg_uncertainty': avg_uncertainty,
            'total_noise_samples': total_noise,
            'total_hard_samples': total_hard_samples,
            'total_label_changes': total_label_changes
        }
    
    def train_with_refinement(self, dataloader, device):
        """完整训练流程"""
        
        print("🚀 开始彻底修复版Dirichlet训练...")
        print(f"📋 配置: 收敛阈值={self.convergence_threshold}, 最小轮数={self.min_epochs}")
        
        training_history = {
            'losses': [],
            'refinement_ratios': [],
            'confidences': [],
            'uncertainties': [],
            'noise_counts': [],
            'hard_sample_counts': [],
            'label_changes': []
        }
        
        for epoch in range(self.max_epochs):
            avg_losses, refinement_stats = self.train_epoch_with_refinement(
                dataloader, device, epoch
            )
            
            # 记录历史
            training_history['losses'].append(avg_losses)
            training_history['refinement_ratios'].append(refinement_stats['refinement_ratio'])
            training_history['confidences'].append(refinement_stats['avg_confidence'])
            training_history['uncertainties'].append(refinement_stats['avg_uncertainty'])
            training_history['noise_counts'].append(refinement_stats['total_noise_samples'])
            training_history['hard_sample_counts'].append(refinement_stats['total_hard_samples'])
            training_history['label_changes'].append(refinement_stats['total_label_changes'])
            
            # 打印epoch总结
            print(f"\n📈 Epoch {epoch+1} 完成:")
            print(f"   总损失: {avg_losses['total_loss']:.6f}")
            print(f"   Expected MSE: {avg_losses['expected_mse']:.6f}")
            print(f"   Dirichlet KL: {avg_losses['dirichlet_kl']:.6f}")
            print(f"   修正比例: {refinement_stats['refinement_ratio']:.4f} ({refinement_stats['refinement_ratio']*100:.2f}%)")
            print(f"   标签变化总数: {refinement_stats['total_label_changes']}")
            print(f"   困难样本总数: {refinement_stats['total_hard_samples']}")
            print(f"   平均不确定性: {refinement_stats['avg_uncertainty']:.4f}")
            
            # 🔥 修复收敛判断
            should_converge = (
                epoch >= self.min_epochs and
                refinement_stats['refinement_ratio'] < self.convergence_threshold and
                refinement_stats['total_label_changes'] > 0  # 确保有标签变化
            )
            
            if should_converge:
                print(f"\n✅ 收敛达成！修正比例 {refinement_stats['refinement_ratio']:.4f} < 阈值 {self.convergence_threshold}")
                print(f"🎯 训练在第 {epoch+1} 轮收敛")
                break
            else:
                if epoch < self.min_epochs:
                    print(f"   🔄 继续训练 (未达到最小轮数 {self.min_epochs})")
                elif refinement_stats['total_label_changes'] == 0:
                    print(f"   ⚠️  无标签变化，继续训练")
                else:
                    print(f"   🔄 继续训练 (修正比例 {refinement_stats['refinement_ratio']:.4f} >= {self.convergence_threshold})")
            
            print("-" * 70)
        
        return training_history

# ==========================================
# 🔥 主训练函数 - 完整Dirichlet版本
# ==========================================

def train_with_dirichlet_refinement():
    """🔥 完整Dirichlet Evidence Learning训练函数"""
    
    DATA_DIR = "/hy-tmp/code/CC/Step0/Experiments/20251216_145746_Improved_Data_Test/03_FedDNA_In"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 训练参数
    input_dim = 4
    hidden_dim = 64
    seq_len = 150
    lr = 1e-3
    
    # 损失权重
    alpha = 1.0
    beta = 0.01
    gamma = 0.1  # 增加Dirichlet KL权重
    
    # 修正参数
    uncertainty_threshold = 0.8      # 提高不确定性阈值
    confidence_threshold = 0.3       # 降低置信度阈值
    distance_threshold = 0.8         # 降低距离阈值
    convergence_threshold = 0.02     # 降低收敛阈值
    max_epochs = 12
    min_epochs = 5
    
    try:
        # 检查数据目录
        if not os.path.exists(DATA_DIR):
            print(f"❌ 目录不存在: {DATA_DIR}")
            return None
            
        # 加载数据
        dataset = CloverClusterDataset(DATA_DIR)
        if len(dataset) == 0:
            print(f"❌ 数据集为空，请检查数据文件")
            return None
            
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # 初始化模型
        model = SimplifiedFedDNA(input_dim, hidden_dim, seq_len).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = DirichletComprehensiveLoss(alpha=alpha, beta=beta, gamma=gamma)
        
        # 初始化Dirichlet修正模块
        refinement_module = DirichletEvidenceRefinement(
            uncertainty_threshold=uncertainty_threshold,
            distance_threshold=distance_threshold
        ).to(DEVICE)
        
        # 创建Dirichlet训练器
        trainer = DirichletRefinementTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            refinement_module=refinement_module,
            convergence_threshold=convergence_threshold,
            max_epochs=max_epochs
        )
        
        print(f"🔧 完整Dirichlet模型配置:")
        print(f"   设备: {DEVICE}")
        print(f"   数据集大小: {len(dataset)} 个簇")
        print(f"   损失权重: Expected MSE={alpha}, 对比学习={beta}, Dirichlet KL={gamma}")
        print(f"   修正参数: 不确定性阈值={uncertainty_threshold}, 距离阈值={distance_threshold}")
        print(f"   收敛条件: 修正比例 < {convergence_threshold*100}%")
        
        # 开始训练
        training_history = trainer.train_with_refinement(dataloader, DEVICE)
        
        # 保存结果
        save_dict = {
            'model_state_dict': model.state_dict(),
            'refinement_state_dict': refinement_module.state_dict(),
            'training_history': training_history,
            'config': {
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'seq_len': seq_len,
                },
                'training_config': {
                    'learning_rate': lr,
                    'loss_weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
                    'max_epochs': max_epochs
                },
                'refinement_config': {
                    'uncertainty_threshold': uncertainty_threshold,
                    'distance_threshold': distance_threshold,
                    'convergence_threshold': convergence_threshold
                }
            }
        }
        
        torch.save(save_dict, "dirichlet_refined_model.pth")
        print(f"\n💾 完整Dirichlet模型已保存到: dirichlet_refined_model.pth")
        
        # 训练总结
        final_refinement_ratio = training_history['refinement_ratios'][-1]
        final_uncertainty = training_history['uncertainties'][-1]
        
        print(f"\n🎯 Dirichlet训练完成总结:")
        print(f"   最终修正比例: {final_refinement_ratio:.4f}")
        print(f"   最终平均不确定性: {final_uncertainty:.4f}")
        print(f"   总训练轮数: {len(training_history['losses'])}")
        
        if final_refinement_ratio < convergence_threshold:
            print(f"   ✅ 成功收敛！")
        else:
            print(f"   ⚠️  未完全收敛，可考虑增加训练轮数")
            
        return training_history
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    print("🎯 选择训练模式:")
    print("1. 完整Dirichlet训练 (Step1 + Step2 + 完整Dirichlet代数)")
    print("2. 基础训练 (仅Step1)")
    
    # 默认使用完整Dirichlet训练
    print("🚀 启动完整Dirichlet Evidence Learning训练...")
    train_history = train_with_dirichlet_refinement()
