#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNA 聚类 Metadata 生成脚本 - 修复张量维度问题
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

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
# 核心组件（保持不变）
# ==========================================

class SimpleEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, seq_len=150):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.conmamba = ConmambaBlock(dim=hidden_dim)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conmamba(x)
        return x

class ContrastiveLearning(nn.Module):
    def __init__(self, hidden_dim=64, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, embeddings):
        seq_repr = torch.mean(embeddings, dim=1)
        projected = self.projection(seq_repr)
        return F.normalize(projected, dim=-1)

class DirichletEvidenceDecoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=4):
        super().__init__()
        self.output_dim = output_dim
        
        self.evidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )
        
    def forward(self, x):
        evidence = self.evidence_net(x)
        alpha = evidence + 1.0
        alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)
        predictions = alpha / alpha_sum
        K = self.output_dim
        uncertainty = K / alpha_sum.squeeze(-1)
        evidence_strength = torch.sum(evidence, dim=-1)
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'predictions': predictions,
            'uncertainty': uncertainty,
            'strength': evidence_strength
        }

class DirichletEvidenceFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, dirichlet_outputs):
        evidence = dirichlet_outputs['evidence']
        uncertainty = dirichlet_outputs['uncertainty']
        
        fusion_weights = 1.0 / (uncertainty + 1e-8)
        fusion_weights = fusion_weights.unsqueeze(-1)
        
        weighted_evidence = evidence * fusion_weights
        fused_evidence = torch.sum(weighted_evidence, dim=0)
        total_weights = torch.sum(fusion_weights, dim=0)
        
        fused_evidence = fused_evidence / (total_weights + 1e-8)
        
        fused_alpha = fused_evidence + 1.0
        fused_alpha_sum = torch.sum(fused_alpha, dim=-1, keepdim=True)
        fused_predictions = fused_alpha / fused_alpha_sum
        fused_uncertainty = evidence.shape[-1] / fused_alpha_sum.squeeze(-1)
        
        return {
            'fused_evidence': fused_evidence,
            'fused_alpha': fused_alpha,
            'fused_predictions': fused_predictions,
            'fused_uncertainty': fused_uncertainty,
            'fusion_weights': fusion_weights.squeeze(-1)
        }

class SimplifiedFedDNA(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, seq_len=150):
        super().__init__()
        self.encoder = SimpleEncoder(input_dim, hidden_dim, seq_len)
        self.contrastive = ContrastiveLearning(hidden_dim)
        self.evidence_decoder = DirichletEvidenceDecoder(hidden_dim, input_dim)
        self.evidence_fusion = DirichletEvidenceFusion()
        
    def forward(self, reads_batch):
        B, N, L, D = reads_batch.shape
        reads_flat = reads_batch.view(B * N, L, D)
        
        embeddings = self.encoder(reads_flat)
        contrastive_features = self.contrastive(embeddings)
        dirichlet_outputs = self.evidence_decoder(embeddings)
        
        for key in dirichlet_outputs:
            if dirichlet_outputs[key].dim() == 3:
                dirichlet_outputs[key] = dirichlet_outputs[key].view(B, N, L, -1)
            elif dirichlet_outputs[key].dim() == 2:
                dirichlet_outputs[key] = dirichlet_outputs[key].view(B, N, L)
        
        contrastive_features = contrastive_features.view(B, N, -1)
        
        return dirichlet_outputs, contrastive_features

class DirichletComprehensiveLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01, temperature=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        
    def dirichlet_expected_mse(self, fused_predictions, target):
        mse = torch.mean((fused_predictions - target) ** 2)
        return mse
    
    def dirichlet_kl_divergence(self, alpha, target_alpha=None):
        if target_alpha is None:
            K = alpha.shape[-1]
            target_alpha = torch.ones_like(alpha)
        
        alpha_sum = torch.sum(alpha, dim=-1, keepdim=True)
        target_alpha_sum = torch.sum(target_alpha, dim=-1, keepdim=True)
        
        kl_div = (
            torch.lgamma(alpha_sum) - torch.lgamma(target_alpha_sum) +
            torch.sum(torch.lgamma(target_alpha) - torch.lgamma(alpha), dim=-1, keepdim=True) +
            torch.sum((alpha - target_alpha) * (torch.digamma(alpha) - torch.digamma(alpha_sum)), dim=-1, keepdim=True)
        )
        
        return torch.mean(kl_div)
    
    def contrastive_loss(self, features, cluster_labels=None):
        if features.shape[0] <= 1:
            return torch.tensor(0.0, device=features.device)
            
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
        fused_predictions = fusion_results['fused_predictions']
        fused_alpha = fusion_results['fused_alpha']
        
        expected_mse = self.dirichlet_expected_mse(fused_predictions, target)
        contrastive_loss = self.contrastive_loss(contrastive_features)
        dirichlet_kl = self.dirichlet_kl_divergence(fused_alpha)
        
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
# 🔥 修复版修正模块 - 解决张量维度问题
# ==========================================

class DirichletEvidenceRefinementFixed(nn.Module):
    def __init__(self, 
                 uncertainty_threshold_start=0.4,
                 uncertainty_threshold_end=0.3,
                 confidence_threshold_start=0.3,
                 confidence_threshold_end=0.6,
                 distance_threshold=2.0,
                 max_refinement_ratio=0.5,
                 force_refinement_ratio=0.1):
        super().__init__()
        self.uncertainty_threshold_start = uncertainty_threshold_start
        self.uncertainty_threshold_end = uncertainty_threshold_end
        self.confidence_threshold_start = confidence_threshold_start
        self.confidence_threshold_end = confidence_threshold_end
        self.distance_threshold = distance_threshold
        self.max_refinement_ratio = max_refinement_ratio
        self.force_refinement_ratio = force_refinement_ratio
        
        self.global_cluster_centers = None
        
    def get_adaptive_thresholds(self, epoch, max_epochs):
        progress = min(epoch / max(max_epochs - 1, 1), 1.0)
        uncertainty_threshold = (self.uncertainty_threshold_start + 
                               progress * (self.uncertainty_threshold_end - self.uncertainty_threshold_start))
        confidence_threshold = (self.confidence_threshold_start + 
                              progress * (self.confidence_threshold_end - self.confidence_threshold_start))
        return uncertainty_threshold, confidence_threshold
    
    def calculate_dirichlet_confidence(self, uncertainty):
        avg_uncertainty = torch.mean(uncertainty, dim=1)
        confidence_scores = torch.exp(-1.0 * avg_uncertainty)
        return confidence_scores, avg_uncertainty
    
    def update_global_centers(self, embeddings, labels):
        if embeddings.shape[0] > 1:
            unique_labels = torch.unique(labels[labels != -1])
            if len(unique_labels) > 1:
                device = embeddings.device
                feature_dim = embeddings.shape[1]
                centers = torch.zeros(3, feature_dim, device=device)
                
                for i, label in enumerate(unique_labels):
                    if i >= 3: break
                    mask = (labels == label)
                    if mask.sum() > 0:
                        centers[i] = torch.mean(embeddings[mask], dim=0)
                
                for i in range(len(unique_labels), 3):
                    centers[i] = torch.randn(feature_dim, device=device) * 0.1
                
                self.global_cluster_centers = centers
    
    def create_initial_labels_improved(self, embeddings, num_clusters=3):
        N = embeddings.shape[0]
        device = embeddings.device
        
        if N == 1:
            if self.global_cluster_centers is not None:
                distances = torch.cdist(embeddings, self.global_cluster_centers)
                label = torch.argmin(distances, dim=1)
                return label
            else:
                label = torch.randint(0, num_clusters, (1,), device=device)
                return label
        
        if N <= num_clusters:
            return torch.arange(N, device=device)
        
        try:
            from sklearn.cluster import KMeans
            embeddings_np = embeddings.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_np)
            return torch.tensor(labels, device=device, dtype=torch.long)
        except ImportError:
            center_indices = torch.randperm(N, device=device)[:num_clusters]
            centers = embeddings[center_indices]
            distances = torch.cdist(embeddings, centers)
            labels = torch.argmin(distances, dim=1)
            return labels
    
    def identify_hard_samples(self, uncertainty, confidence_scores, 
                             uncertainty_threshold, confidence_threshold, N):
        avg_uncertainty = torch.mean(uncertainty, dim=1)
        
        if N == 1:
            single_uncertainty = avg_uncertainty[0].item()
            single_confidence = confidence_scores[0].item()
            should_modify = (single_uncertainty > 0.35) or (single_confidence < 0.65)
            hard_sample_mask = torch.tensor([should_modify], device=uncertainty.device)
            
            return hard_sample_mask, {
                'avg_uncertainty': single_uncertainty,
                'uncertainty_threshold_used': uncertainty_threshold,
                'confidence_threshold_used': confidence_threshold,
            }
        
        high_uncertainty_mask = avg_uncertainty > uncertainty_threshold
        low_confidence_mask = confidence_scores < confidence_threshold
        hard_sample_mask = high_uncertainty_mask | low_confidence_mask
        
        if hard_sample_mask.sum() == 0:
            num_force = max(1, int(N * self.force_refinement_ratio))
            _, worst_indices = torch.topk(avg_uncertainty, num_force)
            hard_sample_mask = torch.zeros(N, dtype=torch.bool, device=uncertainty.device)
            hard_sample_mask[worst_indices] = True
        
        if hard_sample_mask.sum() > N * self.max_refinement_ratio:
            num_hard = int(N * self.max_refinement_ratio)
            hard_indices = torch.where(hard_sample_mask)[0]
            hard_uncertainties = avg_uncertainty[hard_indices]
            _, selected_indices = torch.topk(hard_uncertainties, num_hard)
            new_hard_mask = torch.zeros(N, dtype=torch.bool, device=uncertainty.device)
            new_hard_mask[hard_indices[selected_indices]] = True
            hard_sample_mask = new_hard_mask
        
        return hard_sample_mask, {
            'avg_uncertainty': avg_uncertainty.mean().item(),
            'uncertainty_threshold_used': uncertainty_threshold,
            'confidence_threshold_used': confidence_threshold,
        }
    
    def reassign_single_sample(self, embedding, current_label, num_clusters=3):
        """🔥 修复版单样本重分配 - 返回标量值"""
        device = embedding.device
        current_label_value = current_label.item() if current_label.dim() > 0 else current_label
        
        if self.global_cluster_centers is not None:
            distances = torch.cdist(embedding.unsqueeze(0), self.global_cluster_centers)
            nearest_label = torch.argmin(distances).item()
            
            if nearest_label != current_label_value:
                print(f"       🎯 单样本重分配: {current_label_value} → {nearest_label}")
                return nearest_label, distances.min().item()
            else:
                if torch.rand(1).item() < 0.3:
                    available_labels = [i for i in range(num_clusters) if i != current_label_value]
                    if available_labels:
                        new_label = np.random.choice(available_labels)
                        print(f"       🎲 单样本随机改变: {current_label_value} → {new_label}")
                        return new_label, distances.min().item()
        
        if torch.rand(1).item() < 0.2:
            available_labels = [i for i in range(num_clusters) if i != current_label_value]
            if available_labels:
                new_label = np.random.choice(available_labels)
                print(f"       🎲 单样本随机改变: {current_label_value} → {new_label}")
                return new_label, 1.0
        
        print(f"       ✅ 单样本保持: {current_label_value}")
        return current_label_value, 0.0
    
    def compute_cluster_centers(self, embeddings, labels, num_clusters):
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
    
    def reassign_hard_samples(self, hard_embeddings, cluster_centers, 
                            hard_indices, current_labels):
        if hard_embeddings.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=hard_embeddings.device), torch.tensor([], device=hard_embeddings.device)
            
        distances = torch.cdist(hard_embeddings, cluster_centers)
        min_distances, nearest_clusters = torch.min(distances, dim=1)
        
        current_hard_labels = current_labels[hard_indices]
        new_labels = nearest_clusters.clone()
        
        current_distances = distances[torch.arange(len(hard_indices)), current_hard_labels]
        improvement_ratio = (current_distances - min_distances) / (current_distances + 1e-8)
        
        # 宽松策略：只要有1%改善就改变
        significant_improvement = improvement_ratio > 0.01
        
        # 对没有改善的样本，50%概率随机改变
        no_improvement_mask = ~significant_improvement
        if no_improvement_mask.sum() > 0:
            random_change = torch.rand(no_improvement_mask.sum(), device=hard_embeddings.device) < 0.5
            # 保持最近邻分配（已经在new_labels中）
        
        noise_mask = min_distances > self.distance_threshold
        new_labels[noise_mask] = -1
        
        changed_count = (new_labels != current_hard_labels).sum().item()
        print(f"       🔧 多样本重分配: {changed_count}/{len(hard_indices)} 个标签改变")
        
        return new_labels, min_distances
    
    def forward(self, embeddings, dirichlet_uncertainty, current_labels, num_clusters, 
                epoch=0, max_epochs=10):
        """🔥 修复版主流程 - 解决张量维度问题"""
        N = embeddings.shape[0]
        device = embeddings.device
        
        uncertainty_threshold, confidence_threshold = self.get_adaptive_thresholds(epoch, max_epochs)
        
        if N > 1:
            self.update_global_centers(embeddings, current_labels)
        
        unique_labels = torch.unique(current_labels)
        if len(unique_labels) == 1:
            print(f"    🔄 重新初始化标签...")
            current_labels = self.create_initial_labels_improved(embeddings, num_clusters)
        
        confidence_scores, avg_uncertainty = self.calculate_dirichlet_confidence(dirichlet_uncertainty)
        
        hard_sample_mask, criteria_stats = self.identify_hard_samples(
            dirichlet_uncertainty, confidence_scores, uncertainty_threshold, confidence_threshold, N
        )
        
        high_confidence_mask = ~hard_sample_mask
        new_labels = current_labels.clone()
        
        reassignment_stats = {'label_change_count': 0}
        
        if hard_sample_mask.sum() > 0:
            print(f"    🔧 处理 {hard_sample_mask.sum().item()} 个困难样本...")
            
            if N == 1:
                # 🔥 修复单样本处理
                hard_embedding = embeddings[0]
                current_label = current_labels[0]
                new_label_value, distance = self.reassign_single_sample(hard_embedding, current_label, num_clusters)
                
                # 🔥 正确赋值：直接使用标量值
                new_labels[0] = new_label_value
                reassignment_stats['label_change_count'] = 1 if new_label_value != current_label.item() else 0
                
            else:
                # 多样本处理
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
                
                reassigned_labels, distances = self.reassign_hard_samples(
                    hard_embeddings, cluster_centers, hard_indices, current_labels
                )
                
                old_hard_labels = new_labels[hard_sample_mask].clone()
                new_labels[hard_sample_mask] = reassigned_labels
                reassignment_stats['label_change_count'] = (reassigned_labels != old_hard_labels).sum().item()
        
        total_label_changes = (new_labels != current_labels).sum().item()
        
        return new_labels, {
            'total_samples': N,
            'hard_samples_count': hard_sample_mask.sum().item(),
            'label_changes': total_label_changes,
            'refinement_ratio': total_label_changes / N if N > 0 else 0.0,
            'avg_confidence': confidence_scores.mean().item(),
            'avg_uncertainty': avg_uncertainty.mean().item(),
            **criteria_stats,
            **reassignment_stats
        }

# ==========================================
# 数据集（保持不变）
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
# 训练器（保持不变）
# ==========================================

class DirichletRefinementTrainerFixed:
    def __init__(self, model, criterion, optimizer, refinement_module, 
                 convergence_threshold=0.05, max_epochs=15, min_epochs=5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.refinement = refinement_module
        self.convergence_threshold = convergence_threshold
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.persistent_labels = {}
        
    def train_epoch_with_refinement(self, dataloader, device, epoch):
        self.model.train()
        epoch_losses = {'total_loss': 0, 'expected_mse': 0, 'contrastive_loss': 0, 'dirichlet_kl': 0}
        all_refinement_stats = []
        step_count = 0
        
        print(f"\n🔄 Epoch {epoch+1}...")
        
        for i, (reads, ref) in enumerate(dataloader):
            reads = reads.to(device)
            ref = ref.squeeze(0).to(device)
            N = reads.shape[1]
            
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
            
            with torch.no_grad():
                batch_key = f"batch_{i}"
                if batch_key not in self.persistent_labels:
                    initial_labels = torch.randint(0, 3, (N,), device=device)
                    self.persistent_labels[batch_key] = initial_labels
                
                current_labels = self.persistent_labels[batch_key]
                dirichlet_uncertainty = single_batch_outputs['uncertainty']
                
                new_labels, refinement_stats = self.refinement(
                    embeddings=contrastive_features_flat,
                    dirichlet_uncertainty=dirichlet_uncertainty,
                    current_labels=current_labels,
                    num_clusters=3,
                    epoch=epoch,
                    max_epochs=self.max_epochs
                )
                
                self.persistent_labels[batch_key] = new_labels
                all_refinement_stats.append(refinement_stats)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            step_count += 1
            
            # 每10个batch输出一次进度
            if (i + 1) % 10 == 0:
                print(f"  📊 Step {i+1:3d} | Loss: {losses['total_loss'].item():.4f} | "
                      f"修正: {refinement_stats['refinement_ratio']:.3f}")
        
        avg_losses = {key: val / max(1, step_count) for key, val in epoch_losses.items()}
        
        if all_refinement_stats:
            avg_refinement_ratio = np.mean([s['refinement_ratio'] for s in all_refinement_stats])
            avg_confidence = np.mean([s['avg_confidence'] for s in all_refinement_stats])
            avg_uncertainty = np.mean([s['avg_uncertainty'] for s in all_refinement_stats])
            total_hard_samples = sum([s['hard_samples_count'] for s in all_refinement_stats])
            total_label_changes = sum([s['label_changes'] for s in all_refinement_stats])
        else:
            avg_refinement_ratio = 0.0
            avg_confidence = 0.0
            avg_uncertainty = 0.0
            total_hard_samples = 0
            total_label_changes = 0
            
        return avg_losses, {
            'refinement_ratio': avg_refinement_ratio,
            'avg_confidence': avg_confidence,
            'avg_uncertainty': avg_uncertainty,
            'total_hard_samples': total_hard_samples,
            'total_label_changes': total_label_changes
        }
    
    def train_with_refinement(self, dataloader, device):
        print("🚀 开始训练...")
        training_history = {
            'losses': [],
            'refinement_ratios': [],
            'uncertainties': []
        }
        
        for epoch in range(self.max_epochs):
            avg_losses, refinement_stats = self.train_epoch_with_refinement(dataloader, device, epoch)
            
            training_history['losses'].append(avg_losses)
            training_history['refinement_ratios'].append(refinement_stats['refinement_ratio'])
            training_history['uncertainties'].append(refinement_stats['avg_uncertainty'])
            
            print(f"\n📈 Epoch {epoch+1} 完成:")
            print(f"   损失: {avg_losses['total_loss']:.4f}")
            print(f"   修正比例: {refinement_stats['refinement_ratio']:.4f} ({refinement_stats['refinement_ratio']*100:.1f}%)")
            print(f"   标签变化: {refinement_stats['total_label_changes']}")
            print(f"   不确定性: {refinement_stats['avg_uncertainty']:.4f}")
            print(f"   置信度: {refinement_stats['avg_confidence']:.4f}")
            
            if (epoch >= self.min_epochs and 
                refinement_stats['refinement_ratio'] < self.convergence_threshold):
                print(f"\n✅ 收敛！修正比例 {refinement_stats['refinement_ratio']:.4f} < {self.convergence_threshold}")
                break
            else:
                if epoch < self.min_epochs:
                    print(f"   🔄 继续训练 (未达到最小轮数 {self.min_epochs})")
                else:
                    print(f"   🔄 继续训练 (修正比例 {refinement_stats['refinement_ratio']:.4f} >= {self.convergence_threshold})")
            
            print("-" * 60)
        
        return training_history

# ==========================================
# 主函数
# ==========================================

def train_with_fixed_dirichlet_refinement():
    print("🚀 开始修复版Dirichlet训练...")
    
    DATA_DIR = "CC/Step0/Experiments/20251216_145746_Improved_Data_Test/03_FedDNA_In"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    dataset = CloverClusterDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"📦 数据加载完成: {len(dataset)} 个样本")
    
    model = SimplifiedFedDNA(input_dim=4, hidden_dim=128, seq_len=150).to(device)
    criterion = DirichletComprehensiveLoss(alpha=1.0, beta=0.1, gamma=0.01, temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    refinement_module = DirichletEvidenceRefinementFixed(
        uncertainty_threshold_start=0.4,
        uncertainty_threshold_end=0.3,
        confidence_threshold_start=0.3,
        confidence_threshold_end=0.6,
        distance_threshold=2.0,
        max_refinement_ratio=0.5,
        force_refinement_ratio=0.1
    )
    
    trainer = DirichletRefinementTrainerFixed(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        refinement_module=refinement_module,
        convergence_threshold=0.05,
        max_epochs=10,
        min_epochs=3
    )
    
    training_history = trainer.train_with_refinement(dataloader, device)
    
    model_save_path = "fixed_dirichlet_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history,
        'persistent_labels': trainer.persistent_labels,
    }, model_save_path)
    
    print(f"\n💾 模型已保存到: {model_save_path}")
    
    return model, training_history

if __name__ == "__main__":
    model, history = train_with_fixed_dirichlet_refinement()
