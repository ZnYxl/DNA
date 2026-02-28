# models/step1_model.py
"""
SSI-EC 证据学习模型 (Universal Edition)

修复清单:
  [FIX-P0]  self_reconstruction_loss: target 从 inputs（噪声read）改为 consensus_target（伪reference）
  [FIX-P0]  forward: 接收 consensus_target 并传入重建损失

保持不变:
  - encoder/decoder 结构
  - 对比学习 (InfoNCE + Memory Queue)
  - KL 正则
  - 不确定性分解 decompose_uncertainty
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Model import Encoder, RNNBlock
from utils.Loss import CEBayesRiskLoss, KLDivergenceLoss
import numpy as np


# ---------------------------------------------------------------------------
# Masked Bayes Risk Loss
# 屏蔽 Padding 位置（全 0 行）的梯度，防止 Padding 污染不确定性估计
# ---------------------------------------------------------------------------
def masked_bayes_risk(evidence, consensus_target):
    """
    Args:
        evidence:         (B, L, 4) 模型输出的证据
        consensus_target: (B, L, 4) 伪 reference 的 one-hot 编码
                          Padding 位置为全 0 行 [0,0,0,0]
    Returns:
        scalar loss，只在有效碱基位置计算，Padding 位置贡献为 0
    """
    # Padding 位置：consensus_target 的行全为 0（因为 seq_to_onehot 对超长部分 zero-pad）
    # 有效位置：该行至少有一个 1
    mask = consensus_target.sum(dim=-1) > 0  # (B, L)，有效位为 True

    alpha = evidence + 1.0                         # (B, L, 4)
    S = alpha.sum(dim=-1, keepdim=True)            # (B, L, 1)

    # CEBayesRisk: Σ_c y_c * (ψ(S) - ψ(α_c))，逐位置计算
    per_pos_loss = (consensus_target * (
        torch.digamma(S) - torch.digamma(alpha)
    )).sum(dim=-1)                                 # (B, L)

    # 只保留有效位置
    masked_loss = per_pos_loss * mask.float()      # (B, L)

    # 除以真实有效长度，而非固定的 max_length
    valid_len = mask.float().sum(dim=-1).clamp(min=1.0)  # (B,)
    loss_per_seq = masked_loss.sum(dim=-1) / valid_len   # (B,)

    return loss_per_seq.mean()


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
                 queue_size=8192,
                 tau_sim=0.1,
                 tau_weight=1.0,
                 r_ale=2.0,
                 cl_mode='ours'):
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

        self.tau_sim    = tau_sim
        self.tau_weight = tau_weight
        self.r_ale      = r_ale      # 噪声惩罚指数 (1-max_U_ale)^r，越大惩罚越陡
        self.cl_mode    = cl_mode    # 消融实验模式: 'standard'|'ale_only'|'epi_only'|'ours'

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
        ptr = int(self.queue_ptr.item())

        # B > queue_size 时只保留最后 queue_size 条，避免 remain > queue_size 崩溃
        if B >= self.queue_size:
            self.queue_z.copy_(proj_emb[-self.queue_size:].detach())
            self.queue_u_epi.copy_(u_epi[-self.queue_size:].detach().unsqueeze(1))
            self.queue_u_ale.copy_(u_ale[-self.queue_size:].detach().unsqueeze(1))
            self.queue_labels.copy_(labels[-self.queue_size:].detach())
            self.queue_ptr[0] = 0
            self.queue_count[0] = self.queue_size
            return

        if ptr + B <= self.queue_size:
            self.queue_z[ptr:ptr+B] = proj_emb.detach()
            self.queue_u_epi[ptr:ptr+B] = u_epi.detach().unsqueeze(1)
            self.queue_u_ale[ptr:ptr+B] = u_ale.detach().unsqueeze(1)
            self.queue_labels[ptr:ptr+B] = labels.detach()
        else:
            space = self.queue_size - ptr
            self.queue_z[ptr:] = proj_emb[:space].detach()
            self.queue_u_epi[ptr:] = u_epi[:space].detach().unsqueeze(1)
            self.queue_u_ale[ptr:] = u_ale[:space].detach().unsqueeze(1)
            self.queue_labels[ptr:] = labels[:space].detach()
            remain = B - space          # remain < queue_size，因为上面已处理 B >= queue_size
            self.queue_z[:remain] = proj_emb[space:].detach()
            self.queue_u_epi[:remain] = u_epi[space:].detach().unsqueeze(1)
            self.queue_u_ale[:remain] = u_ale[space:].detach().unsqueeze(1)
            self.queue_labels[:remain] = labels[space:].detach()

        self.queue_ptr[0] = (ptr + B) % self.queue_size
        self.queue_count[0] = min(self.queue_count[0] + B, self.queue_size)

    # ------------------------------------------------------------------
    # Encoder + Decoder
    # ------------------------------------------------------------------
    def encode_reads(self, reads):
        # FedDNA Encoder 期望输入 (B, L, 4)，内部会 unsqueeze(1) 当 2D 图像处理
        # 不做 permute，直接传入
        embeddings = self.encoder(reads)  # (B, L, dim)

        if self.length_adapter is not None:
            embeddings = self.length_adapter(embeddings.permute(0, 2, 1)).permute(0, 2, 1)

        pooled_emb = embeddings.mean(dim=1)  # (B, dim)
        return embeddings, pooled_emb

    def decode_to_evidence(self, embeddings):
        # RNNBlock: LSTM(dim→256) → Linear(256→4) → softplus
        # 输出已经是 (B, L, 4)，直接使用
        output = self.rnnblock(embeddings)   # (B, L, 4)
        evidence = F.softplus(output)        # (B, L, 4)，确保非负

        alpha = evidence + 1                 # Dirichlet 参数
        strength = alpha.sum(dim=-1)         # (B, L)，Dirichlet concentration
        return evidence, strength, alpha

    # ------------------------------------------------------------------
    # 对比学习
    # ------------------------------------------------------------------
    def _pair_weight(self, ue_i, ue_j, ua_i, ua_j):
        """
        消融实验统一入口，通过 cl_mode 控制权重公式：

          'standard' — 标准 InfoNCE，w=1（无加权）
          'ale_only' — 只用偶然不确定性惩罚噪声对
          'epi_only' — 只用认知不确定性降低难样本权重
          'ours'     — 完整设计（论文方法）
        """
        if self.cl_mode == 'standard':
            return torch.ones_like(ue_i)

        elif self.cl_mode == 'ale_only':
            return (1.0 - torch.max(ua_i, ua_j)).clamp(min=0.0) ** self.r_ale

        elif self.cl_mode == 'epi_only':
            return torch.exp(-(ue_i + ue_j) / self.tau_weight)

        else:  # 'ours'（默认）
            epi_term = torch.exp(-(ue_i + ue_j) / self.tau_weight)
            ale_term = (1.0 - torch.max(ua_i, ua_j)).clamp(min=0.0) ** self.r_ale
            return epi_term * ale_term

    def uncertainty_weighted_contrastive(self, pooled_emb, cluster_labels, u_epi, u_ale):
        """
        不确定性加权对比损失（论文设计版）

        与标准 InfoNCE 的三处关键差异:
          1. 权重分离 U_epi / U_ale 语义，不再混加
          2. 正样本不加权（分子无 w），负样本加权（分母中 w_neg·exp）
          3. loss 按 Σw_{i,pos} 加权平均，而非简单 mean

        公式:
          L = - (1/Σw_{i,pos}) Σ_i w_{i,pos} · log[
                    exp(s_{i,pos}/τ)
                  / (exp(s_{i,pos}/τ) + Σ_{k∈N_i} w_{ik}·exp(s_{ik}/τ))
              ]
        """
        proj_emb = F.normalize(self.projection_head(pooled_emb), dim=-1)
        B = proj_emb.shape[0]

        # ── In-batch 相似度 + mask ─────────────────────────────────────────
        sim_inbatch  = torch.matmul(proj_emb, proj_emb.T) / self.tau_sim  # (B, B)
        labels_row   = cluster_labels.unsqueeze(1)
        labels_col   = cluster_labels.unsqueeze(0)
        self_mask    = torch.eye(B, dtype=torch.bool, device=self.device)
        pos_mask_inbatch = (labels_row == labels_col) & ~self_mask         # (B, B)

        # ── In-batch pair 权重 ────────────────────────────────────────────
        ue_r = u_epi.unsqueeze(1).expand(B, B)
        ue_c = u_epi.unsqueeze(0).expand(B, B)
        ua_r = u_ale.unsqueeze(1).expand(B, B)
        ua_c = u_ale.unsqueeze(0).expand(B, B)
        w_inbatch = self._pair_weight(ue_r, ue_c, ua_r, ua_c)             # (B, B)

        # ── Queue ─────────────────────────────────────────────────────────
        if int(self.queue_count.item()) > 0:
            q_count  = int(self.queue_count.item())
            q_z      = self.queue_z[:q_count].detach().clone()             # (Q, 128)
            q_labels = self.queue_labels[:q_count].detach().clone()
            q_ue     = self.queue_u_epi[:q_count].detach().clone().squeeze(1)  # (Q,)
            q_ua     = self.queue_u_ale[:q_count].detach().clone().squeeze(1)

            sim_queue    = torch.matmul(proj_emb, q_z.T) / self.tau_sim   # (B, Q)
            pos_mask_q   = (cluster_labels.unsqueeze(1) == q_labels.unsqueeze(0))

            ue_r_q = u_epi.unsqueeze(1).expand(B, q_count)
            ua_r_q = u_ale.unsqueeze(1).expand(B, q_count)
            w_queue = self._pair_weight(
                ue_r_q, q_ue.unsqueeze(0).expand(B, q_count),
                ua_r_q, q_ua.unsqueeze(0).expand(B, q_count)
            )                                                              # (B, Q)

            sim_full      = torch.cat([sim_inbatch, sim_queue],  dim=1)   # (B, B+Q)
            w_full        = torch.cat([w_inbatch,   w_queue],    dim=1)
            pos_mask_full = torch.cat([pos_mask_inbatch, pos_mask_q], dim=1)
        else:
            sim_full      = sim_inbatch
            w_full        = w_inbatch
            pos_mask_full = pos_mask_inbatch

        # ── Loss（正样本不加权，负样本加权）──────────────────────────────
        # 数值稳定：减去每行最大值
        sim_stable = sim_full - sim_full.max(dim=1, keepdim=True)[0].detach()
        exp_sim    = torch.exp(sim_stable)                                 # (B, B+Q)

        neg_mask_full = ~pos_mask_full & ~torch.cat(
            [self_mask, torch.zeros(B, sim_full.shape[1]-B,
                                    dtype=torch.bool, device=self.device)], dim=1
        )

        # 分子：正样本 exp（不加权）
        pos_exp_sum = (exp_sim * pos_mask_full.float()).sum(dim=1)         # (B,)
        # 分母：正样本 exp + 加权负样本 exp
        neg_exp_sum = (exp_sim * w_full * neg_mask_full.float()).sum(dim=1)
        denominator = pos_exp_sum + neg_exp_sum + 1e-10
        numerator   = pos_exp_sum + 1e-10
        log_prob    = torch.log(numerator) - torch.log(denominator)        # (B,)

        # 只对 in-batch 有正样本的 read 计算 loss
        has_pos = (pos_mask_inbatch.sum(dim=1) > 0)
        if not has_pos.any():
            self._dequeue_and_enqueue(proj_emb, u_epi, u_ale, cluster_labels)
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}

        # 加权平均：w_{i,pos} = 该 read 对所有 in-batch 正样本的权重之和
        w_pos_per_read = (w_inbatch * pos_mask_inbatch.float()).sum(dim=1)[has_pos].clamp(min=1e-6)
        loss = -(w_pos_per_read * log_prob[has_pos]).sum() / w_pos_per_read.sum()

        # ── 三个诊断探针 ──────────────────────────────────────────────────
        # 探针目的：验证权重是否真的"两极分化"，以及 embedding 是否在分离
        # 使用 u_ale 阈值 0.5 区分干净/脏样本（实际运行中对应 Zone I vs Zone III）
        probe_stats = {}
        with torch.no_grad():
            clean_mask = (u_ale < 0.5)   # 粗略的"干净"判断
            dirty_mask = (u_ale >= 0.5)

            # 探针 A：干净-干净 pair 的平均 w_ij vs 脏-任意 pair 的平均 w_ij
            cc_mask = clean_mask.unsqueeze(1) & clean_mask.unsqueeze(0) & ~self_mask
            dd_mask = dirty_mask.unsqueeze(1) & ~self_mask
            if cc_mask.any():
                probe_stats['w_clean_clean'] = w_inbatch[cc_mask].mean().item()
            if dd_mask.any():
                probe_stats['w_dirty_any']   = w_inbatch[dd_mask].mean().item()

            # 探针 B：正样本 vs 负样本的平均余弦相似度（不除以 τ）
            cos_sim = torch.matmul(proj_emb, proj_emb.T)                  # (B, B)
            if pos_mask_inbatch.any():
                probe_stats['cos_sim_pos'] = cos_sim[pos_mask_inbatch].mean().item()
            neg_mask_inbatch = ~pos_mask_inbatch & ~self_mask
            if neg_mask_inbatch.any():
                probe_stats['cos_sim_neg'] = cos_sim[neg_mask_inbatch].mean().item()

        self._dequeue_and_enqueue(proj_emb, u_epi, u_ale, cluster_labels)
        return loss, probe_stats

    # ------------------------------------------------------------------
    # 重建损失
    # [FIX-P0] target 从 inputs（噪声read） → consensus_target（伪reference）
    # ------------------------------------------------------------------
    def self_reconstruction_loss(self, evidence, alpha, cluster_labels, consensus_target):
        """
        Args:
            evidence:         (B, L, 4) model evidence output
            alpha:            (B, L, 4) Dirichlet parameters
            cluster_labels:   (B,) cluster IDs
            consensus_target: (B, L, 4) one-hot of pseudo-reference (伪 reference)
                              ← [FIX] 原来传入的是 inputs（带噪 read），现在传入 consensus
        """
        kld_loss_fn = KLDivergenceLoss().to(self.device)

        # masked_bayes_risk: 屏蔽 Padding 位置（全 0 行），防止梯度污染
        input_recon_loss = masked_bayes_risk(evidence, consensus_target)

        total_kl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        processed_clusters = 0

        unique_labels = torch.unique(cluster_labels)
        for label in unique_labels:
            if label < 0: continue
            mask = (cluster_labels == label)
            if mask.sum() < 2: continue

            cluster_evidence = evidence[mask]
            weights = F.softmax(
                torch.log(alpha[mask].sum(dim=-1).mean(dim=1) + 1), dim=0
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
            total_kl_loss = total_kl_loss / processed_clusters

        return input_recon_loss, total_kl_loss, {}

    # ------------------------------------------------------------------
    # Forward
    # [FIX-P0] 接收 consensus_target 并传给重建损失
    # ------------------------------------------------------------------
    def forward(self, reads, cluster_labels, consensus_target, epoch=0):
        """
        Args:
            reads:            (B, L, 4) one-hot input reads
            cluster_labels:   (B,) cluster IDs
            consensus_target: (B, L, 4) one-hot pseudo-reference  ← [NEW]
            epoch:            current training epoch
        """
        embeddings, pooled_emb = self.encode_reads(reads)
        evidence, strength, alpha = self.decode_to_evidence(embeddings)
        strength_seq = strength.mean(dim=-1)

        u_epi, u_ale = decompose_uncertainty(alpha)

        con_loss, probe_stats = self.uncertainty_weighted_contrastive(
            pooled_emb, cluster_labels, u_epi, u_ale
        )

        # [FIX-P0] 传入 consensus_target 而非 reads
        recon_loss, kl_loss, _ = self.self_reconstruction_loss(
            evidence, alpha, cluster_labels, consensus_target
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
            # ── 三个诊断探针（每 batch 打印，验证权重两极分化）──────────
            'w_clean_clean':    probe_stats.get('w_clean_clean', float('nan')),
            'w_dirty_any':      probe_stats.get('w_dirty_any',   float('nan')),
            'cos_sim_pos':      probe_stats.get('cos_sim_pos',   float('nan')),
            'cos_sim_neg':      probe_stats.get('cos_sim_neg',   float('nan')),
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