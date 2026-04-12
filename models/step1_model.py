# models/step1_model.py
"""
SSI-EC 证据学习模型 (Universal Edition)

修复清单:
  [FIX-P0]       self_reconstruction_loss: target 从 inputs（噪声read）改为 consensus_target（伪reference）
  [FIX-P0]       forward: 接收 consensus_target 并传入重建损失
  [FIX-SOFT-NEG] uncertainty_weighted_contrastive: 动态软负样本屏蔽
                 在 proj_emb 空间中余弦相似度 > 0.80 的异簇对不参与负样本排斥，
                 消除 Clover 过分割导致的同源碎片排斥力势垒。
                 自适应退火：训练初期 proj_emb 未收敛，屏蔽自然休眠；
                 收敛后同源碎片相似度突破阈值，屏蔽平滑介入。

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
    u_ale = u_ale_per_pos.mean(dim=-1).clamp(min=0.0)

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
                 r_ale=8.0,
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
        emb_dim = dim
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
        evidence = output                    # RNNBlock内部已做Softplus，直接使用

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
        proj_emb = F.normalize(pooled_emb, dim=-1)
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

        # ── [FIX-SOFT-NEG] 动态软负样本屏蔽（自适应退火） ────────────────
        # 物理含义：Clover 过分割导致同一 GT 分子被切成多个子簇，这些子簇在
        # proj_emb 超球面上余弦相似度 > 0.80，却被当成硬负样本互相推开，
        # 形成排斥力势垒，阻止 Step 2 的 MNN 完成合并。
        #
        # 修复：检测 batch 内高相似度的异簇对，从 neg_mask 中移除，卸除排斥力。
        #
        # 自适应退火（Feature, not a Bug）：
        #   训练初期 proj_emb 未收敛 → 同源碎片相似度低于 0.80 → 屏蔽休眠
        #   收敛后同源碎片相似度突破 0.80 → 屏蔽平滑介入
        #   无需手动调度，完全数据驱动。
        #
        # 数学自洽性：作用在 proj_emb（L2 归一化，128 维）上，
        # 与 InfoNCE 梯度所在的超球面空间一致，避免空间撕裂。
        with torch.no_grad():
            # proj_emb 已 L2 归一化，matmul 直接得余弦相似度矩阵
            cos_sim_batch   = torch.matmul(proj_emb, proj_emb.T)         # (B, B)
            diff_label_mask = ~pos_mask_inbatch & ~self_mask             # batch 内异簇且非自身
            high_sim_mask   = (cos_sim_batch) > 0.98  # [v5] 放宽阈值防坍缩) & diff_label_mask  # 高相似度的异簇对

        # [问题2修复] 原来只屏蔽 batch 内列（:B），Queue 列（B:）完全不受影响。
        # 但 Queue 有 8192 条历史 reads，Clover 5.75× 过分割意味着同一分子的碎片
        # 高概率分散在 batch 和 Queue 中，Queue 里的同源碎片照样被当硬负样本推开。
        # 修复：对 Queue 列也计算 batch × queue 的高相似度异簇对，同样屏蔽。
        neg_mask_full[:, :B] &= ~high_sim_mask   # batch 内列屏蔽（原有逻辑）

        # Queue 列屏蔽（新增）
        if sim_full.shape[1] > B:
            q_count = sim_full.shape[1] - B
            with torch.no_grad():
                q_z_local    = self.queue_z[:q_count].detach().clone()           # (Q, 128)
                cos_sim_queue = torch.matmul(proj_emb, q_z_local.T)             # (B, Q)
                q_labels_local = self.queue_labels[:q_count].detach().clone()   # (Q,)
                diff_label_q   = (cluster_labels.unsqueeze(1) !=
                                  q_labels_local.unsqueeze(0))                   # (B, Q)
                high_sim_q     = (cos_sim_queue) > 0.98  # [v5] 放宽阈值防坍缩) & diff_label_q
            neg_mask_full[:, B:] &= ~high_sim_q
            n_soft_masked_q = int(high_sim_q.sum().item())
        else:
            n_soft_masked_q = 0

        n_soft_masked = int(high_sim_mask.sum().item()) + n_soft_masked_q

        # 分子：正样本 exp（不加权）
        pos_exp_sum = (exp_sim * pos_mask_full.float()).sum(dim=1)         # (B,)
        # 分母：正样本 exp + 加权负样本 exp
        neg_exp_sum = (exp_sim * w_full * neg_mask_full.float()).sum(dim=1)
        denominator = pos_exp_sum + neg_exp_sum + 1e-10
        numerator   = pos_exp_sum + 1e-10
        log_prob    = torch.log(numerator) - torch.log(denominator)        # (B,)

        # [问题2修复] has_pos 必须用 pos_mask_full（含Queue），不能用 pos_mask_inbatch。
        # 原来用 inbatch：一条 read 在 batch 内没有同簇样本但 Queue 里有，
        # has_pos=False → 这条 read 的 loss 被丢弃，Queue 正样本白建了。
        # 同时 w_pos_per_read 也必须改用 w_full * pos_mask_full，
        # 否则分子权重来自 inbatch、分母正样本来自 full，数学不对齐。
        has_pos = (pos_mask_full.sum(dim=1) > 0)
        if not has_pos.any():
            self._dequeue_and_enqueue(proj_emb, u_epi, u_ale, cluster_labels)
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}

        w_pos_per_read = (w_full * pos_mask_full.float()).sum(dim=1)[has_pos].clamp(min=1e-6)
        loss = -(w_pos_per_read * log_prob[has_pos]).sum() / w_pos_per_read.sum()

        # ── 诊断探针 ──────────────────────────────────────────────────────
        # 探针目的：验证权重是否真的"两极分化"，以及 embedding 是否在分离
        probe_stats = {}
        probe_stats['soft_neg_masked'] = n_soft_masked   # 软屏蔽计数（监控退火进度）
        with torch.no_grad():
            ale_median = u_ale.median()
            clean_mask = (u_ale < ale_median)
            dirty_mask = (u_ale >= ale_median)

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

            # [G老师-Bug1-FIX] KL Padding mask：防止 Padding 位置的梯度毒化。
            # 问题：Padding 位置 evidence≈0，argmax 默认返回 0（='A'），
            # KL 强制模型在空白区域输出确定的 'A'，扭曲隐空间流形。
            # 修复：Padding 位置 evidence 清零（→ alpha=[1,1,1,1] 即先验），
            # target 设为均匀分布（0.25），KL(先验 || 先验)=0，无梯度。
            cluster_consensus = consensus_target[mask]
            valid_pos = cluster_consensus[0].sum(dim=-1) > 0  # (L,)
            padding_mask = ~valid_pos
            if padding_mask.any():
                cluster_evidence = cluster_evidence.clone()
                cluster_evidence[:, padding_mask, :] = 0.0
                target_one_hot = target_one_hot.clone()
                target_one_hot[:, padding_mask, :] = 0.25

            total_kl_loss = total_kl_loss + kld_loss_fn(cluster_evidence, target_one_hot)
            processed_clusters += 1

        if processed_clusters > 0:
            total_kl_loss = total_kl_loss / processed_clusters

        return input_recon_loss, total_kl_loss, {}

    # ------------------------------------------------------------------
    # Forward
    # [FIX-P0] 接收 consensus_target 并传给重建损失
    # ------------------------------------------------------------------
    def forward(self, reads, cluster_labels, consensus_target, epoch=0, round_idx=1):
        """
        Args:
            reads:            (B, L, 4) one-hot input reads
            cluster_labels:   (B,) cluster IDs
            consensus_target: (B, L, 4) one-hot pseudo-reference  ← [NEW]
            epoch:            current training epoch
            round_idx:        当前迭代轮次，控制annealing_coef逻辑
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

        annealing_coef = min(1.0, max(0.0, (epoch - 5) / 10.0)) if round_idx == 1 else 1.0
        total_loss = con_loss + recon_loss + annealing_coef * 0.05 * kl_loss

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
            # ── 诊断探针（每 batch 打印，验证权重两极分化 + 软屏蔽退火进度）──
            'w_clean_clean':    probe_stats.get('w_clean_clean',   float('nan')),
            'w_dirty_any':      probe_stats.get('w_dirty_any',     float('nan')),
            'cos_sim_pos':      probe_stats.get('cos_sim_pos',     float('nan')),
            'cos_sim_neg':      probe_stats.get('cos_sim_neg',     float('nan')),
            'soft_neg_masked':  probe_stats.get('soft_neg_masked', 0),
        }
        return loss_dict, outputs


# ---------------------------------------------------------------------------
# 预训练加载
# ---------------------------------------------------------------------------
def load_pretrained_feddna(model, path, device, max_length=150):
    """
    加载 FedDNA 预训练权重到 SSI-EC 模型。

    [致命缺陷修复] length_adapter 动态恢复
      原来 model.__init__ 设 self.length_adapter = None，导致 model.state_dict()
      不含 length_adapter 的键，过滤条件 `k in model_sd` 会静默丢弃 checkpoint
      里的 length_adapter 权重。Decoder 收到未经长度变换的特征，预训练权重失效。

      修复：在构建 model_sd 之前，先探测 checkpoint 是否包含 length_adapter，
      如果维度与当前 max_length 兼容就动态实例化，这样 state_dict 过滤时
      能正确匹配并加载权重。

    Args:
        model:      Step1EvidentialModel 实例
        path:       checkpoint 文件路径
        device:     计算设备
        max_length: 当前 SSI-EC 的序列统一长度。length_adapter 的输入输出
                    维度必须都等于 max_length 才能兼容（SSI-EC 中 read 和
                    consensus 都 pad 到 max_length，不存在长度映射需求）。
    """
    try:
        ckpt = torch.load(path, map_location=device)
        sd = ckpt['model'] if 'model' in ckpt else (
            ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        )

        # [核心修复] 动态探测并实例化 length_adapter（在构建 model_sd 之前）
        # FedDNA 的 length_adapter 是 nn.Linear(noise_length, label_length)，
        # weight shape = (label_length, noise_length) = (out_features, in_features)。
        # SSI-EC 中 encoder 输出的序列长度 = max_length，所以 Linear 的输入维度
        # 必须 = max_length；输出也需要 = max_length（consensus 也是 max_length）。
        if 'length_adapter.weight' in sd:
            sh = sd['length_adapter.weight'].shape  # (out_features, in_features)
            if sh[1] == max_length and sh[0] == max_length:
                # 输入输出维度都匹配，可以安全加载
                import torch.nn as nn_
                model.length_adapter = nn_.Linear(sh[1], sh[0]).to(device)
                print(f"   🔧 动态恢复 length_adapter: Linear({sh[1]}, {sh[0]})")
            else:
                # 维度不匹配（FedDNA 的 noise_length/label_length ≠ max_length）
                # 跳过 length_adapter，Round 1 训练会让模型适应没有它的架构
                print(f"   ⚠️ checkpoint 的 length_adapter 维度 {sh} "
                      f"与 max_length={max_length} 不兼容，已跳过")
                print(f"   💡 提示: 若要利用完整预训练权重，可设 --max_length {sh[1]}")

        # 此时如果 length_adapter 被实例化，model.state_dict() 会包含它的键
        model_sd = model.state_dict()
        new_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        model.load_state_dict(new_sd, strict=False)
        print(f"   ✅ 成功加载预训练权重: {len(new_sd)} 层")
    except Exception as e:
        print(f"   ⚠️ 加载权重失败: {e}")
    return model