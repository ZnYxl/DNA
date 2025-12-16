#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧬 无监督证据驱动DNA聚类与纠错模型
核心策略: 迭代自举 (Iterative Self-Refinement)
- 不依赖Ground Truth
- 通过迭代更新consensus逐步逼近真实序列
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 导入FedDNA核心组件
# ==========================================
try:
    from models.conmamba import ConmambaBlock
    print("✅ 成功导入 ConmambaBlock")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ==========================================
# 工具函数
# ==========================================

def safe_log(x, eps=1e-8):
    return torch.log(torch.clamp(x, min=eps))

def safe_div(x, y, eps=1e-8):
    return x / torch.clamp(y, min=eps)

def check_tensor_health(tensor, name="tensor"):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return False
    return True

# ==========================================
# FedDNA核心组件
# ==========================================

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

class Conv2dUpampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_dropout_p: float, kernel_size=3):
        super(Conv2dUpampling, self).__init__()
        padding = calc_same_padding(kernel_size)
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=conv_dropout_p)

    def forward(self, inputs):
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()
        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)
        outputs = self.dropout(outputs)
        return outputs

class FedDNAEncoder(nn.Module):
    def __init__(self, dim: int = 128):
        super(FedDNAEncoder, self).__init__()
        self.dim = dim
        self.upsampling = Conv2dUpampling(in_channels=1, out_channels=dim//4, conv_dropout_p=0.1)
        self.conmamba = ConmambaBlock(
            dim=dim, ff_mult=4, conv_expansion_factor=2, conv_kernel_size=31,
            attn_dropout=0.1, ff_dropout=0.1, conv_dropout=0.1
        )

    def forward(self, x):
        x = self.upsampling(x)
        x = self.conmamba(x)
        return x

class FedDNARNNBlock(nn.Module):
    def __init__(self, in_channels: int, lstm_hidden_dim: int = 256, rnn_dropout_p=0.1):
        super(FedDNARNNBlock, self).__init__()
        self.rnn = nn.LSTM(
            input_size=in_channels, hidden_size=lstm_hidden_dim, 
            num_layers=2, bidirectional=False, batch_first=True
        )
        self.linear = nn.Linear(in_features=lstm_hidden_dim, out_features=4)
        self.dropout = nn.Dropout(rnn_dropout_p)

    def forward(self, input):
        output, _ = self.rnn(input)
        output = self.linear(output)
        output = self.dropout(output)
        return F.softplus(output) + 0.1

def ds_fusion(evidence: torch.Tensor) -> torch.Tensor:
    """证据融合：在reads维度上取平均"""
    fused_evidence = torch.mean(evidence, dim=0)
    return torch.clamp(fused_evidence, min=0.1, max=100.0)

# ==========================================
# 无监督损失函数
# ==========================================

class UnsupervisedConsensusLoss(nn.Module):
    """
    无监督损失函数：不依赖GT
    
    核心思想：
    1. 让模型输出的evidence与当前伪标签一致
    2. 同时鼓励模型输出高置信度的预测
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, evidence, pseudo_label):
        """
        evidence: [L, 4] - 模型融合后的evidence
        pseudo_label: [L, 4] - 当前的伪标签 (one-hot)
        """
        evidence = torch.clamp(evidence, min=0.1, max=100.0)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        
        # 预测概率
        prob = safe_div(alpha, S)
        prob = torch.clamp(prob, min=1e-8, max=1.0)
        
        # 1. 与伪标签的交叉熵
        log_prob = safe_log(prob)
        ce_loss = -torch.sum(pseudo_label * log_prob, dim=-1)
        
        # 2. 置信度奖励：鼓励高置信度预测（低熵）
        entropy = -torch.sum(prob * log_prob, dim=-1)
        
        # 总损失 = 交叉熵 + 熵正则（较小权重）
        total_loss = ce_loss + 0.1 * entropy
        
        return torch.mean(total_loss), torch.mean(ce_loss), torch.mean(entropy)


class ReadConsistencyLoss(nn.Module):
    """
    Reads一致性损失：鼓励同一cluster内的reads产生相似的evidence
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, evidence_per_read):
        """
        evidence_per_read: [N, L, 4] - 每条read的evidence
        """
        N = evidence_per_read.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=evidence_per_read.device)
        
        # 计算每条read的概率分布
        alpha = evidence_per_read + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        prob = alpha / S  # [N, L, 4]
        
        # 计算均值分布
        mean_prob = torch.mean(prob, dim=0, keepdim=True)  # [1, L, 4]
        
        # 每条read与均值的KL散度
        kl_div = torch.sum(prob * (safe_log(prob) - safe_log(mean_prob)), dim=-1)  # [N, L]
        
        return torch.mean(kl_div)

# ==========================================
# 主模型
# ==========================================

class UnsupervisedDNACorrector(nn.Module):
    """无监督DNA纠错模型"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128, seq_len: int = 150):
        super().__init__()
        
        self.dim = hidden_dim
        self.seq_len = seq_len
        
        self.encoder = FedDNAEncoder(dim=hidden_dim)
        self.length_adapter = nn.Linear(seq_len, seq_len)
        self.rnnblock = FedDNARNNBlock(in_channels=hidden_dim, lstm_hidden_dim=256)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, reads_batch, return_per_read=False):
        """
        reads_batch: [N, L, 4] - 单个cluster的N条reads
        
        返回:
            fused_evidence: [L, 4] - 融合后的evidence
            (可选) evidence_per_read: [N, L, 4]
        """
        N, L, D = reads_batch.shape
        
        # 编码
        encoded = self.encoder(reads_batch)  # [N, L, dim]
        
        if not check_tensor_health(encoded, "encoded"):
            safe_output = torch.ones(L, 4, device=reads_batch.device) * 0.25
            if return_per_read:
                return safe_output, torch.ones(N, L, 4, device=reads_batch.device) * 0.25
            return safe_output
        
        # 长度适配
        encoded = encoded.permute(0, 2, 1)
        encoded = self.length_adapter(encoded)
        encoded = encoded.permute(0, 2, 1)
        
        # 生成evidence
        evidence_per_read = self.rnnblock(encoded)  # [N, L, 4]
        
        # 融合
        fused_evidence = ds_fusion(evidence_per_read)  # [L, 4]
        
        if return_per_read:
            return fused_evidence, evidence_per_read
        return fused_evidence
    
    def predict_consensus(self, reads_batch) -> str:
        """从reads预测consensus序列"""
        self.eval()
        with torch.no_grad():
            fused_evidence = self.forward(reads_batch)
            alpha = fused_evidence + 1.0
            prob = alpha / torch.sum(alpha, dim=-1, keepdim=True)
            pred_indices = torch.argmax(prob, dim=-1)
            
            idx_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
            consensus = ''.join([idx_to_base[i.item()] for i in pred_indices])
        return consensus

# ==========================================
# 数据集
# ==========================================

class UnsupervisedDNADataset(Dataset):
    """无监督DNA数据集：伪标签可动态更新"""
    
    def __init__(self, data_dir: str, seq_len: int = 150):
        self.seq_len = seq_len
        self.clusters = []
        self.base_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.idx_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        
        self._load_data(data_dir)
        
    def _load_data(self, data_dir):
        read_path = os.path.join(data_dir, "read.txt")
        ref_path = os.path.join(data_dir, "ref.txt")
        if not os.path.exists(ref_path):
            ref_path = os.path.join(data_dir, "reference.txt")
            
        print(f"📂 加载数据: {data_dir}")
        
        # 加载初始伪标签（聚类中心）
        with open(ref_path, 'r') as f:
            pseudo_labels = [line.strip() for line in f if line.strip()]
        
        # 加载reads
        with open(read_path, 'r') as f:
            content = f.read().strip()
        raw_clusters = content.split("===============================")
        
        for i, cluster_block in enumerate(raw_clusters):
            if not cluster_block.strip() or i >= len(pseudo_labels):
                continue
                
            reads = [r.strip() for r in cluster_block.strip().split('\n') if r.strip()]
            if len(reads) > 0:
                self.clusters.append({
                    'cluster_id': i,
                    'pseudo_label': pseudo_labels[i],  # 可更新的伪标签
                    'reads': reads
                })
                
        print(f"✅ 加载完成: {len(self.clusters)} 个簇")
        
        # 打印初始统计
        self._print_cluster_stats()
    
    def _print_cluster_stats(self):
        """打印cluster统计信息"""
        reads_counts = [len(c['reads']) for c in self.clusters]
        print(f"📊 Cluster统计:")
        print(f"   - 数量: {len(self.clusters)}")
        print(f"   - Reads/cluster: {np.mean(reads_counts):.1f} ± {np.std(reads_counts):.1f}")
        print(f"   - 范围: [{min(reads_counts)}, {max(reads_counts)}]")
    
    def one_hot_encode(self, seq: str) -> np.ndarray:
        arr = np.zeros((self.seq_len, 4), dtype=np.float32)
        for i, char in enumerate(seq[:self.seq_len]):
            if char in self.base_mapping:
                arr[i, self.base_mapping[char]] = 1.0
        return arr
    
    def update_pseudo_label(self, cluster_id: int, new_label: str):
        """更新某个cluster的伪标签"""
        if cluster_id < len(self.clusters):
            self.clusters[cluster_id]['pseudo_label'] = new_label
    
    def get_cluster_data(self, cluster_id: int, num_reads: int = 8):
        """获取单个cluster的数据"""
        if cluster_id >= len(self.clusters):
            return None, None
            
        cluster = self.clusters[cluster_id]
        reads = cluster['reads']
        pseudo_label = cluster['pseudo_label']
        
        # 随机采样reads
        selected_reads = random.sample(reads, min(num_reads, len(reads)))
        
        reads_encoded = [self.one_hot_encode(read) for read in selected_reads]
        label_encoded = self.one_hot_encode(pseudo_label)
        
        return (
            torch.tensor(np.array(reads_encoded)),
            torch.tensor(label_encoded)
        )
    
    def __len__(self):
        return len(self.clusters)
    
    def __getitem__(self, idx):
        return self.get_cluster_data(idx)

# ==========================================
# 🎯 无监督训练器（核心！）
# ==========================================

class UnsupervisedTrainer:
    """
    无监督训练器
    
    核心策略: 迭代自举
    1. 用当前伪标签训练模型
    2. 用模型输出更新伪标签
    3. 重复直到收敛
    """
    
    def __init__(self, 
                 model: UnsupervisedDNACorrector,
                 dataset: UnsupervisedDNADataset,
                 device: torch.device,
                 num_reads: int = 8,
                 lr: float = 1e-4):
        
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.num_reads = num_reads
        
        # 损失函数
        self.consensus_loss = UnsupervisedConsensusLoss()
        self.consistency_loss = ReadConsistencyLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        
        # 训练历史
        self.history = {
            'total_loss': [],
            'ce_loss': [],
            'entropy': [],
            'consistency': [],
            'pseudo_label_change_rate': [],
            'consensus_stability': []
        }
    
    def compute_reads_agreement(self, cluster_id: int) -> float:
        """计算reads与当前伪标签的一致性"""
        cluster = self.dataset.clusters[cluster_id]
        pseudo_label = cluster['pseudo_label']
        reads = cluster['reads']
        
        total_matches = 0
        total_positions = 0
        
        for read in reads:
            min_len = min(len(read), len(pseudo_label))
            matches = sum(r == p for r, p in zip(read[:min_len], pseudo_label[:min_len]))
            total_matches += matches
            total_positions += min_len
        
        return total_matches / max(total_positions, 1)
    
    def update_pseudo_labels(self) -> Tuple[float, float]:
        """
        用模型输出更新所有cluster的伪标签
        
        返回:
            change_rate: 伪标签变化比例
            avg_agreement: 平均一致性
        """
        self.model.eval()
        
        total_changed = 0
        total_positions = 0
        agreements = []
        
        with torch.no_grad():
            for cluster_id in range(len(self.dataset.clusters)):
                cluster = self.dataset.clusters[cluster_id]
                reads = cluster['reads']
                old_label = cluster['pseudo_label']
                
                # 编码所有reads
                reads_encoded = [self.dataset.one_hot_encode(r) for r in reads]
                reads_tensor = torch.tensor(np.array(reads_encoded)).to(self.device)
                
                # 预测新的consensus
                new_label = self.model.predict_consensus(reads_tensor)
                
                # 计算变化
                min_len = min(len(old_label), len(new_label))
                changed = sum(o != n for o, n in zip(old_label[:min_len], new_label[:min_len]))
                total_changed += changed
                total_positions += min_len
                
                # 更新伪标签
                self.dataset.update_pseudo_label(cluster_id, new_label)
                
                # 计算一致性
                agreement = self.compute_reads_agreement(cluster_id)
                agreements.append(agreement)
        
        change_rate = total_changed / max(total_positions, 1)
        avg_agreement = np.mean(agreements)
        
        return change_rate, avg_agreement
    
    def train_epoch(self, epoch: int) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        epoch_stats = {
            'total_loss': 0.0,
            'ce_loss': 0.0,
            'entropy': 0.0,
            'consistency': 0.0
        }
        
        valid_batches = 0
        cluster_indices = list(range(len(self.dataset.clusters)))
        random.shuffle(cluster_indices)
        
        for batch_idx, cluster_id in enumerate(cluster_indices):
            reads, pseudo_label = self.dataset.get_cluster_data(cluster_id, self.num_reads)
            
            if reads is None or len(reads) == 0:
                continue
            
            reads = reads.to(self.device)
            pseudo_label = pseudo_label.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            fused_evidence, evidence_per_read = self.model(reads, return_per_read=True)
            
            # 计算损失
            total_loss, ce_loss, entropy = self.consensus_loss(fused_evidence, pseudo_label)
            consistency = self.consistency_loss(evidence_per_read)
            
            # 总损失
            loss = total_loss + 0.1 * consistency
            
            if not check_tensor_health(loss, "loss"):
                continue
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 记录
            epoch_stats['total_loss'] += loss.item()
            epoch_stats['ce_loss'] += ce_loss.item()
            epoch_stats['entropy'] += entropy.item()
            epoch_stats['consistency'] += consistency.item()
            valid_batches += 1
            
            # 定期输出
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(cluster_indices)} | "
                      f"Loss: {loss.item():.4f} | CE: {ce_loss.item():.4f} | "
                      f"Entropy: {entropy.item():.4f}")
        
        # 平均
        if valid_batches > 0:
            for key in epoch_stats:
                epoch_stats[key] /= valid_batches
        
        return epoch_stats
    
    def train(self, 
              max_epochs: int = 30,
              update_interval: int = 3,
              convergence_threshold: float = 0.01):
        """
        完整训练流程
        
        参数:
            max_epochs: 最大epoch数
            update_interval: 每隔多少epoch更新伪标签
            convergence_threshold: 收敛阈值（伪标签变化率）
        """
        print("=" * 60)
        print("🧬 无监督DNA纠错训练")
        print("=" * 60)
        print(f"策略: 迭代自举 (每{update_interval}个epoch更新伪标签)")
        print(f"收敛条件: 伪标签变化率 < {convergence_threshold * 100}%")
        print("=" * 60)
        
        # 初始一致性
        initial_agreements = [self.compute_reads_agreement(i) for i in range(len(self.dataset))]
        print(f"\n📊 初始状态:")
        print(f"   Reads与伪标签一致性: {np.mean(initial_agreements)*100:.2f}%")
        
        prev_change_rate = 1.0
        
        for epoch in range(max_epochs):
            print(f"\n{'='*60}")
            print(f"🔄 Epoch {epoch + 1}/{max_epochs}")
            print(f"{'='*60}")
            
            # 训练
            epoch_stats = self.train_epoch(epoch)
            
            print(f"\n📈 Epoch {epoch + 1} 结果:")
            print(f"   Total Loss: {epoch_stats['total_loss']:.4f}")
            print(f"   CE Loss: {epoch_stats['ce_loss']:.4f}")
            print(f"   Entropy: {epoch_stats['entropy']:.4f}")
            print(f"   Consistency: {epoch_stats['consistency']:.4f}")
            
            # 记录历史
            self.history['total_loss'].append(epoch_stats['total_loss'])
            self.history['ce_loss'].append(epoch_stats['ce_loss'])
            self.history['entropy'].append(epoch_stats['entropy'])
            self.history['consistency'].append(epoch_stats['consistency'])
            
            # 更新伪标签
            if (epoch + 1) % update_interval == 0:
                print(f"\n🔄 更新伪标签...")
                change_rate, avg_agreement = self.update_pseudo_labels()
                
                self.history['pseudo_label_change_rate'].append(change_rate)
                self.history['consensus_stability'].append(1.0 - change_rate)
                
                print(f"   伪标签变化率: {change_rate*100:.2f}%")
                print(f"   Reads一致性: {avg_agreement*100:.2f}%")
                
                # 收敛检查
                if change_rate < convergence_threshold:
                    print(f"\n✅ 收敛！伪标签变化率 {change_rate*100:.2f}% < {convergence_threshold*100}%")
                    break
                
                # 改进检查
                if change_rate < prev_change_rate:
                    print(f"   📈 伪标签趋于稳定 ({prev_change_rate*100:.2f}% → {change_rate*100:.2f}%)")
                prev_change_rate = change_rate
            
            self.scheduler.step()
        
        # 最终评估
        print(f"\n{'='*60}")
        print("🎉 训练完成!")
        print(f"{'='*60}")
        
        final_agreements = [self.compute_reads_agreement(i) for i in range(len(self.dataset))]
        print(f"📊 最终状态:")
        print(f"   Reads与consensus一致性: {np.mean(final_agreements)*100:.2f}%")
        print(f"   (提升: {(np.mean(final_agreements) - np.mean(initial_agreements))*100:.2f}%)")
        
        return self.history

# ==========================================
# 评估函数
# ==========================================

def evaluate_correction_quality(dataset: UnsupervisedDNADataset, 
                                 ground_truth_path: str = None) -> Dict:
    """
    评估纠错质量
    
    如果有ground truth，计算与真实序列的一致性
    否则，计算内部一致性指标
    """
    results = {
        'internal_consistency': [],
        'gt_accuracy': [] if ground_truth_path else None
    }
    
    # 内部一致性：reads与consensus的匹配度
    for cluster in dataset.clusters:
        consensus = cluster['pseudo_label']
        reads = cluster['reads']
        
        matches = []
        for read in reads:
            min_len = min(len(read), len(consensus))
            match_rate = sum(r == c for r, c in zip(read[:min_len], consensus[:min_len])) / min_len
            matches.append(match_rate)
        
        results['internal_consistency'].append(np.mean(matches))
    
    # 如果有ground truth
    if ground_truth_path and os.path.exists(ground_truth_path):
        # 加载GT
        gt_refs = {}
        with open(ground_truth_path, 'r') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    cluster_id = parts[1]
                    ref_seq = parts[2]
                    if cluster_id not in gt_refs:
                        gt_refs[cluster_id] = ref_seq
        
        # 计算与GT的一致性
        for i, cluster in enumerate(dataset.clusters):
            consensus = cluster['pseudo_label']
            cluster_id = str(i)
            
            if cluster_id in gt_refs:
                gt = gt_refs[cluster_id]
                min_len = min(len(consensus), len(gt))
                accuracy = sum(c == g for c, g in zip(consensus[:min_len], gt[:min_len])) / min_len
                results['gt_accuracy'].append(accuracy)
    
    return results

# ==========================================
# 可视化
# ==========================================

def plot_training_history(history: Dict, output_path: str = "unsupervised_training.png"):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Unsupervised DNA Correction Training', fontsize=14, fontweight='bold')
    
    epochs = range(1, len(history['total_loss']) + 1)
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['total_loss'], 'b-', label='Total Loss', linewidth=2)
    ax1.plot(epochs, history['ce_loss'], 'r--', label='CE Loss', linewidth=2)
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 熵
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['entropy'], 'g-', linewidth=2)
    ax2.set_title('Prediction Entropy (Lower = More Confident)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Entropy')
    ax2.grid(True, alpha=0.3)
    
    # 3. 伪标签变化率
    ax3 = axes[1, 0]
    if history['pseudo_label_change_rate']:
        update_epochs = [i * 3 for i in range(1, len(history['pseudo_label_change_rate']) + 1)]
        ax3.bar(update_epochs, history['pseudo_label_change_rate'], alpha=0.7, color='orange')
        ax3.axhline(y=0.01, color='r', linestyle='--', label='Convergence Threshold')
        ax3.set_title('Pseudo-Label Change Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Change Rate')
        ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 一致性
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['consistency'], 'purple', linewidth=2)
    ax4.set_title('Reads Consistency Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Consistency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 图表已保存: {output_path}")

# ==========================================
# 主函数
# ==========================================

def main():
    print("=" * 60)
    print("🧬 无监督DNA纠错模型")
    print("=" * 60)
    
    # 配置
    DATA_DIR = "CC/Step0/Experiments/20251216_145746_Improved_Data_Test/03_FedDNA_In"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 加载数据
    dataset = UnsupervisedDNADataset(DATA_DIR, seq_len=150)
    
    # 创建模型
    model = UnsupervisedDNACorrector(
        input_dim=4,
        hidden_dim=128,
        seq_len=150
    )
    
    # 尝试加载预训练权重
    pretrained_path = "step1_model.pth"
    if os.path.exists(pretrained_path):
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"✅ 加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"⚠️ 加载失败: {e}")
    
    # 创建训练器
    trainer = UnsupervisedTrainer(
        model=model,
        dataset=dataset,
        device=device,
        num_reads=8,
        lr=1e-4
    )
    
    # 训练
    history = trainer.train(
        max_epochs=30,
        update_interval=3,  # 每3个epoch更新伪标签
        convergence_threshold=0.01  # 变化率<1%则收敛
    )
    
    # 保存结果
    print("\n💾 保存结果...")
    
    # 保存纠错后的consensus
    with open("consensus_corrected.txt", 'w') as f:
        for cluster in dataset.clusters:
            f.write(cluster['pseudo_label'] + '\n')
    print("✅ 纠错结果: consensus_corrected.txt")
    
    # 绘制训练曲线
    plot_training_history(history, "unsupervised_training.png")
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"unsupervised_model_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, model_path)
    print(f"✅ 模型已保存: {model_path}")
    
    # 评估
    print("\n📊 最终评估:")
    results = evaluate_correction_quality(dataset)
    print(f"   内部一致性: {np.mean(results['internal_consistency'])*100:.2f}%")
    
    return model, dataset, history

if __name__ == "__main__":
    model, dataset, history = main()
