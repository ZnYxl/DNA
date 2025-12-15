"""
Step1 模型定义
包含：Encoder, Decoder, 对比学习模块, 证据融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedFedDNA(nn.Module):
    """Step1: 简化的FedDNA模型"""
    
    def __init__(self, input_dim=4, hidden_dim=64, seq_len=150):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Encoder: 特征提取
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder: 证据生成
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # 证据融合模块
        self.evidence_fusion = EvidenceFusion()
        
    def forward(self, reads):
        """
        前向传播
        reads: [batch_size, N, seq_len, input_dim]
        """
        batch_size, N, seq_len, input_dim = reads.shape
        
        # 重塑为 [batch_size * N * seq_len, input_dim]
        reads_flat = reads.view(-1, input_dim)
        
        # Encoder: 特征提取
        features = self.encoder(reads_flat)  # [batch_size * N * seq_len, hidden_dim]
        
        # Decoder: 证据生成
        evidence = self.decoder(features)    # [batch_size * N * seq_len, input_dim]
        
        # 重塑回原始形状
        evidence = evidence.view(batch_size, N, seq_len, input_dim)
        
        # 对比学习特征 (每个read的平均特征)
        contrastive_features = features.view(batch_size, N, seq_len, -1).mean(dim=2)
        
        return evidence, contrastive_features

class EvidenceFusion(nn.Module):
    """证据融合模块"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, evidence_batch):
        """
        证据融合
        evidence_batch: [N, seq_len, input_dim] - N条reads的证据
        """
        N, seq_len, input_dim = evidence_batch.shape
        
        # 计算证据强度 (每个位置的熵)
        epsilon = 1e-8
        log_evidence = torch.log(evidence_batch + epsilon)
        entropy = -torch.sum(evidence_batch * log_evidence, dim=-1)  # [N, seq_len]
        strengths = 1.0 / (1.0 + entropy)  # 转换为强度分数
        
        # 归一化权重
        weights = F.softmax(strengths, dim=0)  # [N, seq_len]
        
        # 加权融合
        fused_evidence = torch.sum(
            evidence_batch * weights.unsqueeze(-1), dim=0
        )  # [seq_len, input_dim]
        
        return fused_evidence, strengths
