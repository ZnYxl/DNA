# models/step1_model.py
"""
SSI-EC 证据学习模型 (Universal Edition)

保持原有架构不变，无严重 Bug。
注:
  - InfoNCE 正对求和方式与标准 SupCon 略有不同 (中等优先级, 不影响正确性)
  - length_adapter 懒初始化对统一长度数据集安全
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Model import Encoder, RNNBlock
from utils.Loss import CEBayesRiskLoss, KLDivergenceLoss
import numpy as np


# ---------------------------------------------------------------------------
# A. 不确定性分解模块
# ---------------------------------------------------------------------------
def decompose_uncertainty(alpha):
    """FedDNA Eq.8 / Eq.9"""
    S = alpha.sum(dim=-1, keepdim=True)
    rho = alpha / S

    psi_alpha_plus1 = torch.digamma(alpha + 1)
    psi_S_plus1 = torch.digamma(S + 1)

    term1 = (rho * (psi_alpha_plus1 - psi_S_plus1)).sum(dim=-1)
    log_rho = torch.log(rho.clamp(min=1e-10))
    term2 = -(rho * log_rho).sum(dim=-1)
    u_epi_per_pos = term1 + term2
    u_epi = u_epi_per_pos.mean(dim=-1).clamp(min=0.0)

    u_ale_per_pos = (rho * (psi_S_plus1 - psi_alpha_plus1)).sum(dim=-1)
    u_ale = u_ale_per_pos.mean(dim=-1).clamp(min=0.0, max=0.95)

    return u_epi, u_ale


class Step1EvidentialModel(nn.Module):
    def __init__(self,
                 dim=256,
                 max_length=150,
                 num_clusters=50,
                 device='cuda',
                 queue_size=128,
                 tau_sim=0.1,
                 tau_weight=1.0):
        super().__init__()

        self.encoder = Encoder(dim=dim)
        self.length_adapter = None
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

        self.tau_sim = tau_sim
        self.tau_weight = tau_weight

        # Memory Queue
        emb_dim = 128
        self.queue_size = queue_size

        self.register_buffer('queue_z',     torch.randn(queue_size, emb_dim))
        self.register_buffer('queue_u_epi', torch.zeros(queue_size, 1))
        self.register_buffer('queue_u_ale', torch.zeros(queue_size, 1))
        self.register_buffer('queue_labels', torch.full((queue_size,), -1, dtype=torch.long))
        self.register_buffer('queue_ptr',   torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_count', torch.zeros(1, dtype=torch.long))

        self.queue_z.copy_(F.normalize(torch.randn(queue_size, emb_dim), dim=-1))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, proj_emb, u_epi, u_ale, labels):
        B = proj_emb.shape[0]
        ptr = int(self.queue_ptr)

        if B > self.queue_size:
            proj_emb = proj_emb[:self.queue_size]
            u_epi    = u_epi[:self.queue_size]
            u_ale    = u_ale[:self.queue_size]
            labels   = labels[:self.queue_size]
            B = self.queue_size

        end = ptr + B
        if end <= self.queue_size:
            self.queue_z[ptr:end]     = proj_emb.detach()
            self.queue_u_epi[ptr:end] = u_epi.detach().unsqueeze(-1)
            self.queue_u_ale[ptr:end] = u_ale.detach().unsqueeze(-1)
            self.queue_labels[ptr:end]= labels.detach()
        else:
            first = self.queue_size - ptr
            self.queue_z[ptr:]        = proj_emb[:first].detach()
            self.queue_u_epi[ptr:]    = u_epi[:first].detach().unsqueeze(-1)
            self.queue_u_ale[ptr:]    = u_ale[:first].detach().unsqueeze(-1)
            self.queue_labels[ptr:]   = labels[:first].detach()

            remain = B - first
            self.queue_z[:remain]     = proj_emb[first:].detach()
            self.queue_u_epi[:remain] = u_epi[first:].detach().unsqueeze(-1)
            self.queue_u_ale[:remain] = u_ale[first:].detach().unsqueeze(-1)
            self.queue_labels[:remain]= labels[first:].detach()
            end = remain

        self.queue_ptr[0]   = end % self.queue_size
        self.queue_count[0] = min(int(self.queue_count) + B, self.queue_size)

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
        evidence = torch.clamp(evidence, min=1e-8, max=1e8)
        alpha    = evidence + 1.0
        strength = torch.sum(alpha, dim=-1)
        return evidence, strength, alpha

    # ------------------------------------------------------------------
    # 不确定性感知对比损失
    # ------------------------------------------------------------------
    def uncertainty_weighted_contrastive(self, pooled_emb, cluster_labels, u_epi, u_ale):
        B = pooled_emb.size(0)
        if B < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        proj_emb = self.projection_head(pooled_emb)
        proj_emb = F.normalize(proj_emb, dim=-1)

        u_epi = u_epi.view(-1)
        u_ale = u_ale.view(-1)

        logits_inbatch = torch.matmul(proj_emb, proj_emb.T) / self.tau_sim
        eye_mask = torch.eye(B, dtype=torch.bool, device=self.device)
        logits_inbatch = logits_inbatch.masked_fill(eye_mask, -1e9)

        epi_sum = u_epi.unsqueeze(1) + u_epi.unsqueeze(0)
        w_exp   = torch.exp(-epi_sum / self.tau_weight)
        ale_max = torch.max(u_ale.unsqueeze(1), u_ale.unsqueeze(0))
        w_ale   = (1.0 - ale_max).clamp(min=0.0)
        w_inbatch = w_exp * w_ale

        Q = int(self.queue_count.item())
        use_queue = (Q > 0)
        logits_queue = None
        w_queue = None

        if use_queue:
            q_z     = self.queue_z[:Q].clone()
            q_u_epi = self.queue_u_epi[:Q, 0].clone()
            q_u_ale = self.queue_u_ale[:Q, 0].clone()

            logits_queue = torch.matmul(proj_emb, q_z.T) / self.tau_sim
            epi_sum_q = u_epi.unsqueeze(1) + q_u_epi.unsqueeze(0)
            w_exp_q   = torch.exp(-epi_sum_q / self.tau_weight)
            ale_max_q = torch.max(u_ale.unsqueeze(1), q_u_ale.unsqueeze(0))
            w_ale_q   = (1.0 - ale_max_q).clamp(min=0.0)
            w_queue   = w_exp_q * w_ale_q

        if use_queue:
            logits_full  = torch.cat([logits_inbatch, logits_queue], dim=1)
            weights_full = torch.cat([w_inbatch, w_queue], dim=1)
        else:
            logits_full  = logits_inbatch
            weights_full = w_inbatch

        labels_col = cluster_labels.unsqueeze(1)
        pos_mask_inbatch = (labels_col == labels_col.T).float()
        pos_mask_inbatch = pos_mask_inbatch.masked_fill(eye_mask, 0.0)

        if use_queue:
            pos_mask_queue = torch.zeros(B, Q, device=self.device)
            pos_mask_full  = torch.cat([pos_mask_inbatch, pos_mask_queue], dim=1)
        else:
            pos_mask_full = pos_mask_inbatch

        logits_max, _ = torch.max(logits_full, dim=1, keepdim=True)
        logits_full_stable = logits_full - logits_max.detach()
        exp_logits = torch.exp(logits_full_stable) * weights_full

        denominator = exp_logits.sum(dim=1) + 1e-10
        numerator = (exp_logits * pos_mask_full).sum(dim=1) + 1e-10
        log_prob = torch.log(numerator) - torch.log(denominator)

        has_pos = (pos_mask_inbatch.sum(dim=1) > 0)
        if has_pos.any():
            loss = -log_prob[has_pos].mean()
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self._dequeue_and_enqueue(proj_emb, u_epi, u_ale, cluster_labels)
        return loss

    # ------------------------------------------------------------------
    # 重建损失
    # ------------------------------------------------------------------
    def self_reconstruction_loss(self, evidence, alpha, cluster_labels, inputs):
        bayes_risk  = CEBayesRiskLoss().to(self.device)
        kld_loss_fn = KLDivergenceLoss().to(self.device)

        input_recon_loss = bayes_risk(evidence, inputs)
        total_kl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        processed_clusters = 0

        unique_labels = torch.unique(cluster_labels)
        for label in unique_labels:
            if label < 0: continue
            mask = (cluster_labels == label)
            if mask.sum() < 2: continue

            cluster_evidence = evidence[mask]
            weights = F.softmax(
                torch.sum(alpha[mask], dim=-1).mean(dim=1), dim=0
            ).view(-1, 1, 1)
            fused_evidence = torch.sum(
                cluster_evidence * weights, dim=0, keepdim=True
            ).detach()

            target_one_hot = F.one_hot(
                fused_evidence.argmax(dim=-1), num_classes=4
            ).float().expand(mask.sum(), -1, -1)

            total_kl_loss = total_kl_loss + kld_loss_fn(cluster_evidence, target_one_hot)
            processed_clusters += 1

        if processed_clusters > 0:
            total_kl_loss /= processed_clusters

        return input_recon_loss, total_kl_loss, {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, reads, cluster_labels, epoch=0):
        embeddings, pooled_emb = self.encode_reads(reads)
        evidence, strength, alpha = self.decode_to_evidence(embeddings)
        strength_seq = strength.mean(dim=-1)

        u_epi, u_ale = decompose_uncertainty(alpha)

        con_loss = self.uncertainty_weighted_contrastive(
            pooled_emb, cluster_labels, u_epi, u_ale
        )
        recon_loss, kl_loss, _ = self.self_reconstruction_loss(
            evidence, alpha, cluster_labels, reads
        )

        annealing_coef = min(1.0, max(0.0, (epoch - 5) / 10.0))
        total_loss = con_loss + 10.0 * recon_loss + annealing_coef * 0.05 * kl_loss

        loss_dict = {
            'total':           total_loss,
            'contrastive':     con_loss,
            'reconstruction':  recon_loss,
            'kl_divergence':   kl_loss,
            'annealing_coef':  annealing_coef
        }
        outputs = {
            'avg_strength':     strength_seq.mean().item(),
            'high_conf_ratio':  (strength_seq > 10.0).float().mean().item(),
            'u_epi_mean':       u_epi.mean().item(),
            'u_ale_mean':       u_ale.mean().item(),
            'queue_count':      int(self.queue_count.item()),
            'u_epi':            u_epi.detach(),
            'u_ale':            u_ale.detach(),
        }
        return loss_dict, outputs


# ---------------------------------------------------------------------------
# 预训练加载
# ---------------------------------------------------------------------------
def load_pretrained_feddna(model, path, device):
    try:
        ckpt = torch.load(path, map_location=device)
        sd = ckpt['model'] if 'model' in ckpt else (
            ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        )
        model_sd = model.state_dict()
        new_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(new_sd, strict=False)
        print(f"   ✅ 成功加载预训练权重: {len(new_sd)} 层")
    except Exception as e:
        print(f"   ⚠️ 加载权重失败: {e}")
    return model