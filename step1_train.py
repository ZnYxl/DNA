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
    """🔥 渐进式修正：避免过度修正"""
    
    def __init__(self, 
                 uncertainty_threshold_start=0.95,    # 初始严格阈值
                 uncertainty_threshold_end=0.85,      # 最终阈值
                 confidence_threshold_start=0.9,      # 初始严格阈值
                 confidence_threshold_end=0.15,        # 最终阈值
                 distance_threshold=0.3,
                 max_refinement_ratio=0.3):           # 最大修正比例限制
        super().__init__()
        self.uncertainty_threshold_start = uncertainty_threshold_start
        self.uncertainty_threshold_end = uncertainty_threshold_end
        self.confidence_threshold_start = confidence_threshold_start
        self.confidence_threshold_end = confidence_threshold_end
        self.distance_threshold = distance_threshold
        self.max_refinement_ratio = max_refinement_ratio
        
    def get_adaptive_thresholds(self, epoch, max_epochs):
        """🔥 自适应阈值：随训练进程调整"""
        progress = min(epoch / max(max_epochs - 1, 1), 1.0)
        
        uncertainty_threshold = (self.uncertainty_threshold_start + 
                               progress * (self.uncertainty_threshold_end - self.uncertainty_threshold_start))
        
        confidence_threshold = (self.confidence_threshold_start + 
                              progress * (self.confidence_threshold_end - self.confidence_threshold_start))
        
        return uncertainty_threshold, confidence_threshold
    
    def calculate_dirichlet_confidence(self, uncertainty):
        """计算置信度 - 使用更稳定的方法"""
        avg_uncertainty = torch.mean(uncertainty, dim=1)  # [N]
        
        # 使用指数变换，更敏感
        confidence_scores = torch.exp(-2.0 * avg_uncertainty)  # 指数衰减
        
        return confidence_scores, avg_uncertainty
    
    def identify_hard_samples_conservative(self, uncertainty, confidence_scores, 
                                         uncertainty_threshold, confidence_threshold):
        """🔥 保守的困难样本识别"""
        N = uncertainty.shape[0]
        
        avg_uncertainty = torch.mean(uncertainty, dim=1)  # [N]
        
        # 更严格的标准
        high_uncertainty_mask = avg_uncertainty > uncertainty_threshold
        low_confidence_mask = confidence_scores < confidence_threshold
        
        # 不确定性方差 - 只选择最不稳定的10%
        uncertainty_var = torch.var(uncertainty, dim=1)  # [N]
        uncertainty_var_threshold = torch.quantile(uncertainty_var, 0.9)  # 前10%
        high_variance_mask = uncertainty_var > uncertainty_var_threshold
        
        # 🔥 严格标准：必须同时满足高不确定性AND低置信度
        hard_sample_mask = high_uncertainty_mask & low_confidence_mask
        
        # 如果还是太多，进一步限制
        if hard_sample_mask.sum() > N * self.max_refinement_ratio:
            # 选择最不��定的样本
            num_hard = int(N * self.max_refinement_ratio)
            _, worst_indices = torch.topk(avg_uncertainty, num_hard)
            hard_sample_mask = torch.zeros(N, dtype=torch.bool, device=uncertainty.device)
            hard_sample_mask[worst_indices] = True
        
        return hard_sample_mask, {
            'high_uncertainty_count': high_uncertainty_mask.sum().item(),
            'low_confidence_count': low_confidence_mask.sum().item(),
            'high_variance_count': high_variance_mask.sum().item(),
            'avg_uncertainty': avg_uncertainty.mean().item(),
            'uncertainty_threshold_used': uncertainty_threshold,
            'confidence_threshold_used': confidence_threshold,
            'max_allowed_hard_samples': int(N * self.max_refinement_ratio)
        }
    
    def create_multi_cluster_assignment(self, embeddings, uncertainty, num_base_clusters=3):
        """创建多簇分配"""
        N = embeddings.shape[0]
        device = embeddings.device
        
        if N <= num_base_clusters:
            return torch.arange(N, device=device)
        
        # 基于不确定性的简单分组
        avg_uncertainty = torch.mean(uncertainty, dim=1)  # [N]
        
        # 按不确定性分成3组：低、中、高
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
                centers[k] = torch.randn(feature_dim, device=device) * 0.1
                
        return centers
        
    def reassign_hard_samples_conservative(
        self,
        hard_embeddings,
        cluster_centers,
        hard_indices,
        current_labels
    ):
        """
        🔥 真·保守重分配策略（不会卡死，也不会乱改）
        """
        device = hard_embeddings.device
        M = hard_embeddings.shape[0]

        if M == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, device=device)
            )

        # 1️⃣ 计算 hard → 所有簇的距离
        distances = torch.cdist(hard_embeddings, cluster_centers)  # [M, K]

        # 当前标签
        current_hard_labels = current_labels[hard_indices]

        # 当前簇距离
        current_distances = distances[
            torch.arange(M, device=device),
            current_hard_labels
        ]

        # 最近簇
        min_distances, nearest_clusters = torch.min(distances, dim=1)

        # 初始化：默认不改
        new_labels = current_hard_labels.clone()

        # ------------------------------------------------------------------
        # 2️⃣ 改标签的“最低触发条件”
        # ------------------------------------------------------------------

        # (1) 最近簇 ≠ 当前簇
        different_cluster = nearest_clusters != current_hard_labels

        # (2) 距离改善比例（比你原来宽松一点）
        improvement_ratio = (current_distances - min_distances) / (current_distances + 1e-8)

        # ✅ 只要有“明显改善”即可（10% 而不是 20%）
        significant_improvement = improvement_ratio > 0.10

        # (3) 当前簇在“距离排序中很靠后”（而不是第一或第二）
        rank_in_clusters = torch.argsort(distances, dim=1).argsort(dim=1)
        badly_ranked = rank_in_clusters[
            torch.arange(M, device=device),
            current_hard_labels
        ] >= 2   # 当前簇排在第 3 名以后

        # 🔥 最终改标签条件（三者同时）
        change_mask = (
            different_cluster
            & significant_improvement
            & badly_ranked
        )

        new_labels[change_mask] = nearest_clusters[change_mask]

        # ------------------------------------------------------------------
        # 3️⃣ 噪声判定（极端情况才触发）
        # ------------------------------------------------------------------

        # 使用 batch 内自适应阈值，而不是固定 distance_threshold
        noise_threshold = torch.quantile(min_distances, 0.95)

        noise_mask = min_distances > noise_threshold

        # ⚠️ 噪声不会覆盖已经成功修正的样本
        noise_mask = noise_mask & (~change_mask)

        new_labels[noise_mask] = -1

        return new_labels, min_distances

    
    def forward(self, embeddings, dirichlet_uncertainty, current_labels, num_clusters, 
                epoch=0, max_epochs=10):
        """🔥 渐进式修正主流程"""
        N = embeddings.shape[0]
        device = embeddings.device
        
        # 🔥 获取自适应阈值
        uncertainty_threshold, confidence_threshold = self.get_adaptive_thresholds(epoch, max_epochs)
        
        # 如果current_labels都相同，创建多簇初始分配
        if len(torch.unique(current_labels)) == 1:
            current_labels = self.create_multi_cluster_assignment(
                embeddings, dirichlet_uncertainty, num_base_clusters=3
            )
            num_clusters = max(3, len(torch.unique(current_labels)))
        
        # 1️⃣ 计算置信度
        confidence_scores, avg_uncertainty = self.calculate_dirichlet_confidence(dirichlet_uncertainty)
        
        # 2️⃣ 保守的困难样本识别
        hard_sample_mask, criteria_stats = self.identify_hard_samples_conservative(
            dirichlet_uncertainty, confidence_scores, uncertainty_threshold, confidence_threshold
        )
        # 🔍 调试：看看“是否真的没有困难样本”
        if hard_sample_mask.any():
            print(
                "⚠️ Hard sample triggered!",
                "count =", hard_sample_mask.sum().item(),
                "uncertainty =", dirichlet_uncertainty[hard_sample_mask][:5].detach().cpu().numpy(),
                "confidence =", confidence_scores[hard_sample_mask][:5].detach().cpu().numpy()
            )
        else:
            print(
                "[Debug] No hard samples | "
                f"uncertainty mean={dirichlet_uncertainty.mean().item():.4f}, "
                f"max={dirichlet_uncertainty.max().item():.4f} | "
                f"confidence mean={confidence_scores.mean().item():.4f}, "
                f"min={confidence_scores.min().item():.4f}"
            )
        high_confidence_mask = ~hard_sample_mask
        
        # 3️⃣ 保留高置信度样本的标签
        new_labels = current_labels.clone()
        
        # 4️⃣ 保守的困难样本处理
        reassignment_stats = {'reassigned_count': 0, 'noise_count': 0, 'label_change_count': 0}
        
        if hard_sample_mask.sum() > 0:
            if high_confidence_mask.sum() > 0:
                high_conf_embeddings = embeddings[high_confidence_mask]
                high_conf_labels = current_labels[high_confidence_mask]
                cluster_centers = self.compute_cluster_centers(
                    high_conf_embeddings, high_conf_labels, num_clusters
                )
            else:
                cluster_centers = self.compute_cluster_centers(
                    embeddings, current_labels, num_clusters
                )
            
            hard_embeddings = embeddings[hard_sample_mask]
            hard_indices = torch.where(hard_sample_mask)[0]
            
            reassigned_labels, distances = self.reassign_hard_samples_conservative(
                hard_embeddings, cluster_centers, hard_indices, current_labels
            )
            if (reassigned_labels != current_labels[hard_indices]).any():
                print(
                    f"🟡 Label changed at epoch {epoch}:",
                    (reassigned_labels != current_labels[hard_indices]).sum().item()
                )

            old_hard_labels = new_labels[hard_sample_mask].clone()
            new_labels[hard_sample_mask] = reassigned_labels
            
            reassignment_stats['reassigned_count'] = (reassigned_labels != -1).sum().item()
            reassignment_stats['noise_count'] = (reassigned_labels == -1).sum().item()
            reassignment_stats['label_change_count'] = (reassigned_labels != old_hard_labels).sum().item()
        
        # 5️⃣ 统计信息
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
            'epoch': epoch,
            **criteria_stats,
            **reassignment_stats
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
# 🔥 改进的训练器 - 渐进式收敛
# ==========================================

class DirichletRefinementTrainer:
    """🔥 渐进式训练器 - 完整版"""
    
    def __init__(self, model, criterion, optimizer, refinement_module, 
                 convergence_threshold=0.05, max_epochs=15, min_epochs=8,
                 uncertainty_improvement_threshold=0.02):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.refinement = refinement_module
        self.convergence_threshold = convergence_threshold
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.uncertainty_improvement_threshold = uncertainty_improvement_threshold
        
    def train_epoch_with_refinement(self, dataloader, device, epoch):
        """训练一个epoch"""
        
        self.model.train()
        epoch_losses = {'total_loss': 0, 'expected_mse': 0, 'contrastive_loss': 0, 'dirichlet_kl': 0}
        all_refinement_stats = []
        step_count = 0
        
        print(f"\n🔄 Epoch {epoch+1} - 渐进式Dirichlet训练...")
        
        for i, (reads, ref) in enumerate(dataloader):
            reads = reads.to(device)
            ref = ref.squeeze(0).to(device)
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
            
            # === 步骤2: 渐进式修正 ===
            with torch.no_grad():
                current_labels = torch.randint(0, 3, (N,), device=device)
                dirichlet_uncertainty = single_batch_outputs['uncertainty']
                
                # 🔥 传递epoch信息给修正模块
                new_labels, refinement_stats = self.refinement(
                    embeddings=contrastive_features_flat,
                    dirichlet_uncertainty=dirichlet_uncertainty,
                    current_labels=current_labels,
                    num_clusters=3,
                    epoch=epoch,
                    max_epochs=self.max_epochs
                )
                
                all_refinement_stats.append(refinement_stats)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            step_count += 1
            
            # 详细输出
            if (i + 1) % 5 == 0:
                print(f"  📊 Step {i+1:3d} | Loss: {losses['total_loss'].item():.4f} | "
                      f"修正: {refinement_stats['refinement_ratio']:.3f} | "
                      f"困难样本: {refinement_stats['hard_samples_count']}/{N}")
                print(f"       阈值: U={refinement_stats['uncertainty_threshold_used']:.3f}, "
                      f"C={refinement_stats['confidence_threshold_used']:.3f} | "
                      f"不确定性: {refinement_stats['avg_uncertainty']:.3f}")
        
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
        """🔥 渐进式训练流程 - 完整版"""
        
        print("🚀 开始渐进式Dirichlet训练...")
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
        
        prev_uncertainty = float('inf')
        convergence_count = 0  # 连续满足收敛条件的次数
        
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
            
            # 计算不确定性改善
            uncertainty_improvement = prev_uncertainty - refinement_stats['avg_uncertainty']
            prev_uncertainty = refinement_stats['avg_uncertainty']
            
            # 打印epoch总结
            print(f"\n📈 Epoch {epoch+1} 完成:")
            print(f"   总损失: {avg_losses['total_loss']:.6f}")
            print(f"   Expected MSE: {avg_losses['expected_mse']:.6f}")
            print(f"   Dirichlet KL: {avg_losses['dirichlet_kl']:.6f}")
            print(f"   修正比例: {refinement_stats['refinement_ratio']:.4f} ({refinement_stats['refinement_ratio']*100:.2f}%)")
            print(f"   困难样本: {refinement_stats['total_hard_samples']}")
            print(f"   不确定性: {refinement_stats['avg_uncertainty']:.4f} (改善: {uncertainty_improvement:+.4f})")
            print(f"   置信度: {refinement_stats['avg_confidence']:.4f}")
            
            # 🔥 完整的收敛判断逻辑
            refinement_converged = refinement_stats['refinement_ratio'] < self.convergence_threshold
            uncertainty_stable = abs(uncertainty_improvement) < self.uncertainty_improvement_threshold
            has_label_changes = refinement_stats['total_label_changes'] > 0
            min_epochs_reached = epoch >= self.min_epochs
            
            # 综合收敛条件
            current_converged = (
                min_epochs_reached and 
                refinement_converged and 
                (uncertainty_stable or refinement_stats['avg_uncertainty'] < 0.5)
            )
            
            if current_converged:
                convergence_count += 1
                print(f"   ✅ 满足收敛条件 ({convergence_count}/2)")
            else:
                convergence_count = 0
                if not min_epochs_reached:
                    print(f"   🔄 继续训练 (未达到最小轮数 {self.min_epochs})")
                elif not refinement_converged:
                    print(f"   🔄 继续训练 (修正比例 {refinement_stats['refinement_ratio']:.4f} >= {self.convergence_threshold})")
                elif not uncertainty_stable and refinement_stats['avg_uncertainty'] >= 0.5:
                    print(f"   🔄 继续训练 (不确定性未稳定: {uncertainty_improvement:+.4f})")
            
            # 连续2轮满足收敛条件才真正收敛
            if convergence_count >= 2:
                print(f"\n✅ 收敛达成！连续 {convergence_count} 轮满足收敛条件")
                print(f"🎯 训练在第 {epoch+1} 轮收敛")
                break
            
            # 早停条件：不确定性不再改善且修正比例很小
            if (epoch >= self.min_epochs + 3 and 
                refinement_stats['refinement_ratio'] < 0.01 and 
                abs(uncertainty_improvement) < 0.001):
                print(f"\n🛑 早停：模型已稳定")
                print(f"🎯 训练在第 {epoch+1} 轮早停")
                break
            
            print("-" * 70)
        
        # 🔥 训练完成总结
        final_stats = {
            'total_epochs': epoch + 1,
            'converged': convergence_count >= 2,
            'final_refinement_ratio': refinement_stats['refinement_ratio'],
            'final_uncertainty': refinement_stats['avg_uncertainty'],
            'final_confidence': refinement_stats['avg_confidence'],
            'final_loss': avg_losses['total_loss'],
            'uncertainty_reduction': training_history['uncertainties'][0] - refinement_stats['avg_uncertainty'] if training_history['uncertainties'] else 0,
            'avg_refinement_ratio': np.mean(training_history['refinement_ratios']) if training_history['refinement_ratios'] else 0
        }
        
        print(f"\n🎯 渐进式Dirichlet训练完成总结:")
        print(f"   最终修正比例: {final_stats['final_refinement_ratio']:.4f}")
        print(f"   最终不确定性: {final_stats['final_uncertainty']:.4f}")
        print(f"   最终置信度: {final_stats['final_confidence']:.4f}")
        print(f"   不确定性降低: {final_stats['uncertainty_reduction']:+.4f}")
        print(f"   总训练轮数: {final_stats['total_epochs']}")
        print(f"   平均修正比例: {final_stats['avg_refinement_ratio']:.4f}")
        
        if final_stats['converged']:
            print("   ✅ 成功收敛！")
        else:
            print("   ⚠️  未完全收敛，可考虑:")
            print("      - 增加训练轮数")
            print("      - 调整收敛阈值")
            print("      - 检查数据质量")
        
        return training_history, final_stats

# ==========================================
# 🔥 完整的训练主函数
# ==========================================

def train_with_improved_dirichlet_refinement():
    """🔥 使用改进数据和渐进式训练的完整流程"""
    
    print("🚀 开始改进版Dirichlet Evidence Learning训练...")
    
    # 1️⃣ 数据准备
    print("\n📊 准备训练数据...")
    
    # 🔥 修复：使用默认数据路径
    DATA_DIR = "CC/Step0/Experiments/20251216_145746_Improved_Data_Test/03_FedDNA_In"
    if not os.path.exists(DATA_DIR):
        DATA_DIR = "/hy-tmp/data"  # 备用路径
        print(f"⚠️  使用默认数据集: {DATA_DIR}")
    else:
        print(f"✅ 使用数据集: {DATA_DIR}")
    
    # 2️⃣ 模型和训练组件初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 数据加载
    dataset = CloverClusterDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"📦 数据加载完成: {len(dataset)} 个样本")
    
    # 🔥 修复：使用正确的模型类和参数
    model = SimplifiedFedDNA(
        input_dim=4,
        hidden_dim=128,  # 增加隐藏维度
        seq_len=150
    ).to(device)
    
    # 🔥 修复：使用正确的损失函数类
    criterion = DirichletComprehensiveLoss(
        alpha=1.0,      # Dirichlet Expected MSE权重
        beta=0.1,       # 对比学习损失权重
        gamma=0.01,     # Dirichlet KL散度权重
        temperature=0.1
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    # 🔥 渐进式修正模块
    refinement_module = DirichletEvidenceRefinement(
        uncertainty_threshold_start=0.9,    # 初始严格
        uncertainty_threshold_end=0.7,      # 最终放松
        confidence_threshold_start=0.1,     # 初始严格
        confidence_threshold_end=0.3,       # 最终放松
        distance_threshold=1.0,
        max_refinement_ratio=0.2            # 最大20%修正
    )
    
    # 🔥 渐进式训练器
    trainer = DirichletRefinementTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        refinement_module=refinement_module,
        convergence_threshold=0.08,         # 放宽收敛阈值
        max_epochs=12,
        min_epochs=5,
        uncertainty_improvement_threshold=0.01
    )
    
    # 3️⃣ 开始训练
    print(f"\n🎯 开始渐进式训练...")
    training_history, final_stats = trainer.train_with_refinement(dataloader, device)
    
    # 4️⃣ 保存模型
    model_save_path = "improved_dirichlet_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history,
        'final_stats': final_stats,
        'model_config': {
            'input_dim': 4,
            'hidden_dim': 128,
            'seq_len': 150
        }
    }, model_save_path)
    
    print(f"\n💾 改进版Dirichlet模型已保存到: {model_save_path}")
    
    # 5️⃣ 训练结果可视化（可选）
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 损失曲线
        losses = [l['total_loss'] for l in training_history['losses']]
        axes[0,0].plot(losses)
        axes[0,0].set_title('Training Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        
        # 修正比例
        axes[0,1].plot(training_history['refinement_ratios'])
        axes[0,1].axhline(y=0.08, color='r', linestyle='--', label='Convergence Threshold')
        axes[0,1].set_title('Refinement Ratio')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Ratio')
        axes[0,1].legend()
        
        # 不确定性
        axes[1,0].plot(training_history['uncertainties'])
        axes[1,0].set_title('Average Uncertainty')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Uncertainty')
        
        # 置信度
        axes[1,1].plot(training_history['confidences'])
        axes[1,1].set_title('Average Confidence')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Confidence')
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("📊 训练曲线已保存到: training_curves.png")
        
    except ImportError:
        print("⚠️  matplotlib未安装，跳过可视化")
    
    return model, training_history, final_stats

if __name__ == "__main__":
    model, history, stats = train_with_improved_dirichlet_refinement()
