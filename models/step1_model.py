# models/step1_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Model import Encoder, RNNBlock
from utils.Loss import CEBayesRiskLoss, KLDivergenceLoss
import numpy as np


class Step1EvidentialModel(nn.Module):
    """
    步骤一：Evidence-driven训练模型（修复版）
    ✅ 修复了 Loss 返回字典的 Key，解决日志全0问题
    ✅ 包含 Input Reconstruction 修复
    """

    def __init__(self,
                 dim=256,
                 max_length=150,
                 num_clusters=50,
                 device='cuda'):
        super().__init__()

        self.encoder = Encoder(dim=dim)
        self.length_adapter = None  # 延迟初始化
        self.rnnblock = RNNBlock(in_channels=dim, lstm_hidden_dim=256, rnn_dropout_p=0.1)

        self.projection_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 128)
        )

        self.dim = dim
        self.max_length = max_length
        self.num_clusters = num_clusters
        self.device = device

    def _init_length_adapter_if_needed(self, seq_len):
        if self.length_adapter is None:
            self.length_adapter = nn.Linear(seq_len, self.max_length).to(self.device)

    def encode_reads(self, reads):
        B, L, D = reads.shape
        embeddings = self.encoder(reads)
        pooled_emb = embeddings.mean(dim=1)
        return embeddings, pooled_emb

    def decode_to_evidence(self, embeddings):
        B, L, D = embeddings.shape
        self._init_length_adapter_if_needed(L)

        if L != self.max_length:
            adapted = embeddings.permute(0, 2, 1)
            adapted = self.length_adapter(adapted)
            adapted = adapted.permute(0, 2, 1)
        else:
            adapted = embeddings

        evidence = self.rnnblock(adapted)
        # 限制 evidence 范围，防止梯度爆炸
        evidence = torch.clamp(evidence, min=1e-8, max=1e8)
        alpha = evidence + 1.0
        strength = torch.sum(alpha, dim=-1)
        return evidence, strength, alpha

    def contrastive_learning(self, pooled_emb, cluster_labels, strength, temperature=0.1, epoch=0):
        # 简化版对比学习
        if pooled_emb.size(0) < 2: return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        proj_emb = self.projection_head(pooled_emb)
        proj_emb = F.normalize(proj_emb, dim=-1)
        sim_matrix = torch.matmul(proj_emb, proj_emb.T) / temperature

        labels_expanded = cluster_labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.T).float()
        
        # 自身mask
        logits_mask = torch.scatter(
            torch.ones_like(positive_mask),
            1,
            torch.arange(pooled_emb.size(0)).view(-1, 1).to(self.device),
            0
        )
        positive_mask = positive_mask * logits_mask
        
        exp_sim = torch.exp(torch.clamp(sim_matrix, max=10))
        numerator = torch.sum(exp_sim * positive_mask, dim=1) + 1e-8
        denominator = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim) + 1e-8
        
        # 只有存在正样本对时才计算loss，否则为0
        if positive_mask.sum() > 0:
            return -torch.log(numerator / denominator).mean()
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def self_reconstruction_loss(self, evidence, alpha, cluster_labels, inputs):
        """
        ✅ 核心修复：Input Reconstruction Loss
        """
        bayes_risk = CEBayesRiskLoss().to(self.device)
        kld_loss_fn = KLDivergenceLoss().to(self.device)

        # 1. Input Reconstruction (主 Loss)
        # 强制要求输出等于输入，防止 Mode Collapse
        input_recon_loss = bayes_risk(evidence, inputs)

        # 2. Consensus Consistency (正则项)
        total_kl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        fused_consensus = {}

        unique_labels = torch.unique(cluster_labels)
        processed_clusters = 0

        for label in unique_labels:
            if label < 0: continue
            mask = (cluster_labels == label)
            if mask.sum() < 2: continue

            cluster_evidence = evidence[mask]
            
            # 计算 Consensus Target (Detached, 防止梯度通过target回传)
            weights = F.softmax(torch.sum(alpha[mask], dim=-1).mean(dim=1), dim=0).view(-1, 1, 1)
            fused_evidence = torch.sum(cluster_evidence * weights, dim=0, keepdim=True).detach()
            
            # KL Loss: 让每个read逼近簇中心的one-hot分布
            target_one_hot = F.one_hot(fused_evidence.argmax(dim=-1), num_classes=4).float().expand(mask.sum(), -1, -1)
            total_kl_loss = total_kl_loss + kld_loss_fn(cluster_evidence, target_one_hot)
            processed_clusters += 1

        if processed_clusters > 0:
            total_kl_loss /= processed_clusters

        return input_recon_loss, total_kl_loss, fused_consensus

    def forward(self, reads, cluster_labels, epoch=0):
        embeddings, pooled_emb = self.encode_reads(reads)
        evidence, strength, alpha = self.decode_to_evidence(embeddings)

        # Loss 计算
        con_loss = self.contrastive_learning(pooled_emb, cluster_labels, strength, epoch=epoch)
        
        # ✅ 必须把 reads 传进去
        recon_loss, kl_loss, fused_consensus = self.self_reconstruction_loss(
            evidence, alpha, cluster_labels, reads
        )

        # 退火系数
        annealing_coef = min(1.0, max(0.0, (epoch - 5) / 10.0))
        
        # 总损失 = 1.0 * 对比 + 10.0 * 重建 (Input) + 0.05 * KL
        total_loss = con_loss + 10.0 * recon_loss + annealing_coef * 0.05 * kl_loss

        # ✅ 关键修复：字典 Key 必须与 step1_train.py 中 epoch_losses 的 Key 一致
        loss_dict = {
            'total': total_loss,
            'contrastive': con_loss,       # 对应 epoch_losses['contrastive']
            'reconstruction': recon_loss,  # 对应 epoch_losses['reconstruction']
            'kl_divergence': kl_loss,      # 对应 epoch_losses['kl_divergence']
            'annealing_coef': annealing_coef
        }

        outputs = {
            'avg_strength': strength.mean().item(),
            'high_conf_ratio': (strength.mean(dim=1) > 10.0).float().mean().item()
        }

        return loss_dict, outputs

def load_pretrained_feddna(model, path, device):
    try:
        ckpt = torch.load(path, map_location=device)
        sd = ckpt['model'] if 'model' in ckpt else (ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        
        # 过滤不匹配的键
        model_sd = model.state_dict()
        new_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        
        model.load_state_dict(new_sd, strict=False)
        print(f"   ✅ 成功加载预训练权重: {len(new_sd)} 层")
    except Exception as e:
        print(f"   ⚠️ 加载权重失败: {e}")
    return model