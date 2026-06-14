#!/bin/bash
# =============================================================================
# v20.B Patch 1/4: step1_model.py
#   - 加 topology_regularizer 方法
#   - forward 接收 anchor_centroids (dict[int,Tensor]) 和 lambda_topo (float)
#   - total_loss 加 lambda_topo * L_topo (仅 R≥2)
#   - outputs 加 'topo_loss' 字段
# =============================================================================
set -e

FILE="${1:-/mnt/st_data/liangxinyi/code/models/step1_model.py}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$FILE" ]]; then echo "❌ $FILE 不存在"; exit 1; fi

# ── anchor 检查 ─────────────────────────────────────────────────
ANCHOR_FWD='    def forward(self, reads, cluster_labels, consensus_target, epoch=0, round_idx=1):'
ANCHOR_TOTAL='        total_loss = con_loss + recon_loss + annealing_coef * 0.05 * kl_loss'
ANCHOR_OUT="            'soft_neg_masked':  probe_stats.get('soft_neg_masked', 0),"

for A in "$ANCHOR_FWD" "$ANCHOR_TOTAL" "$ANCHOR_OUT"; do
    n=$(grep -cF "$A" "$FILE" || true)
    if [[ "$n" != "1" ]]; then
        echo "❌ anchor 出现 $n 次: $A"; exit 1
    fi
done
n=$(grep -cF "topology_regularizer\|anchor_centroids\|lambda_topo" "$FILE" || true)
if [[ "$n" != "0" ]]; then echo "⚠️ 已应用过, 跳过"; exit 0; fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] 改动:"
    echo "  + 新增 topology_regularizer 方法"
    echo "  + forward 签名加 anchor_centroids, lambda_topo"
    echo "  + total_loss 加 topo 项"
    echo "  + outputs 加 'topo_loss'"
    exit 0
fi

cp "$FILE" "${FILE}.bak_v20B"
echo "📋 备份: ${FILE}.bak_v20B"

python - "$FILE" << 'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

# ── 改动 1: 新增 topology_regularizer 方法 (插在 forward 之前) ─────
old_fwd = '    def forward(self, reads, cluster_labels, consensus_target, epoch=0, round_idx=1):'
new_method = '''    def topology_regularizer(self, embeddings, cluster_labels, anchor_centroids,
                              min_cluster_size=3):
        """
        [v20B] Cluster topology preservation regularizer.

        物理含义: 防止 contrastive learning 在 R≥2 把同源 cluster 推得更开.
        E_heavy spike 证明: encoder 漂移有方向但有害 (R1→R3 median cos=0.86,
        SR 0.91→0.88). frozen=持平, free=退化, topo regularizer 是滑块.

        L_topo = (1/N) Σ_c [n_c × (1 - cos(c_now, c_anchor))]

        - c_now (带梯度): batch 内 cluster c 的 reads embedding 均值
        - c_anchor (不带梯度): R-1 step2 存档的 centroid
        - n_c (size weight): 大 cluster 主导 loss (centroid 估计更准)
        - min_cluster_size (默认 3): batch 内 reads<3 的 cluster 跳过 (噪声防护)

        Args:
            embeddings:       (B, D) pooled embeddings, 带梯度
            cluster_labels:   (B,) cluster ids
            anchor_centroids: dict {cluster_id (int): Tensor(D,)} 不带梯度
            min_cluster_size: int, 过滤 batch 内 reads 太少的 cluster

        Returns:
            L_topo: scalar tensor (带梯度)
            n_used: int, 实际计算的 cluster 数 (用于诊断)
        """
        if anchor_centroids is None or len(anchor_centroids) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        device = embeddings.device
        unique_cs, inverse = torch.unique(cluster_labels, return_inverse=True)
        L_topo_terms = []
        total_weight = 0
        n_used = 0

        for ci, c in enumerate(unique_cs.tolist()):
            if c < 0:                              # -1 reads 跳过
                continue
            if c not in anchor_centroids:          # anchor 不存在跳过 (新生簇)
                continue
            mask = (cluster_labels == c)
            n_c = int(mask.sum().item())
            if n_c < min_cluster_size:             # 噪声防护
                continue

            c_now = embeddings[mask].mean(dim=0)   # (D,) 带梯度
            c_anchor = anchor_centroids[c].to(device).detach()  # (D,) 不带梯度
            cos_sim = F.cosine_similarity(c_now.unsqueeze(0),
                                          c_anchor.unsqueeze(0)).squeeze()
            L_topo_terms.append(n_c * (1.0 - cos_sim))
            total_weight += n_c
            n_used += 1

        if not L_topo_terms or total_weight == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0

        L_topo = torch.stack(L_topo_terms).sum() / total_weight
        return L_topo, n_used


    def forward(self, reads, cluster_labels, consensus_target, epoch=0, round_idx=1, anchor_centroids=None, lambda_topo=0.0):'''
assert src.count(old_fwd) == 1, f"forward anchor not unique: {src.count(old_fwd)}"
src = src.replace(old_fwd, new_method)

# ── 改动 2: total_loss 加 topo 项 ──────────────────────────────────
old_total = '        total_loss = con_loss + recon_loss + annealing_coef * 0.05 * kl_loss'
new_total = '''        # [v20B] Cluster topology preservation regularizer
        topo_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_topo_used = 0
        if anchor_centroids is not None and lambda_topo > 0 and round_idx >= 2:
            topo_loss, n_topo_used = self.topology_regularizer(
                pooled_emb, cluster_labels, anchor_centroids,
                min_cluster_size=3,
            )

        total_loss = con_loss + recon_loss + annealing_coef * 0.05 * kl_loss + lambda_topo * topo_loss'''
assert src.count(old_total) == 1, f"total_loss anchor not unique: {src.count(old_total)}"
src = src.replace(old_total, new_total)

# ── 改动 3: loss_dict 加 'topo' 字段 ────────────────────────────────
old_lossd = """        loss_dict = {
            'total':           total_loss,
            'contrastive':     con_loss,
            'reconstruction':  recon_loss,
            'kl_divergence':   kl_loss,
            'annealing_coef':  annealing_coef
        }"""
new_lossd = """        loss_dict = {
            'total':           total_loss,
            'contrastive':     con_loss,
            'reconstruction':  recon_loss,
            'kl_divergence':   kl_loss,
            'annealing_coef':  annealing_coef,
            'topo':            topo_loss,            # [v20B]
        }"""
assert src.count(old_lossd) == 1, f"loss_dict anchor not unique: {src.count(old_lossd)}"
src = src.replace(old_lossd, new_lossd)

# ── 改动 4: outputs 加 'topo_loss' / 'n_topo_used' ─────────────────
old_out = "            'soft_neg_masked':  probe_stats.get('soft_neg_masked', 0),"
new_out = """            'soft_neg_masked':  probe_stats.get('soft_neg_masked', 0),
            'topo_loss':       float(topo_loss.item()) if hasattr(topo_loss, 'item') else 0.0,  # [v20B]
            'n_topo_used':     n_topo_used,                                                       # [v20B]"""
assert src.count(old_out) == 1, f"outputs anchor not unique: {src.count(old_out)}"
src = src.replace(old_out, new_out)

open(path, 'w').write(src)
print("✅ step1_model.py patched")
PYEOF

echo
echo "── grep 验证 ──"
grep -n "v20B\|topology_regularizer\|anchor_centroids\|lambda_topo\|topo_loss\|topo\b" "$FILE" | head -15
echo
echo "✅ Patch 1/4 done"