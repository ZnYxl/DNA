#!/bin/bash
# =============================================================================
# v20.A Patch 2/5: step1_model.py
#   - uncertainty_weighted_contrastive 加 kmer_vec 参数
#   - 在 cos>0.98 soft mask 之后, 用 Jaccard 把 cross-cluster 但同源的 pair
#     从 neg_mask 中移除 (保护 anchor)
#   - forward 加 kmer_vec 参数, 转发给 contrastive
# =============================================================================
set -e

FILE="${1:-models/step1_model.py}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$FILE" ]]; then echo "❌ $FILE 不存在"; exit 1; fi

# anchor 检查
ANCHOR_CON='    def uncertainty_weighted_contrastive(self, pooled_emb, cluster_labels, u_epi, u_ale):'
ANCHOR_FWD='    def forward(self, reads, cluster_labels, consensus_target, epoch=0, round_idx=1):'
ANCHOR_CALL='        con_loss, probe_stats = self.uncertainty_weighted_contrastive('

for A in "$ANCHOR_CON" "$ANCHOR_FWD" "$ANCHOR_CALL"; do
    n=$(grep -cF "$A" "$FILE" || true)
    if [[ "$n" != "1" ]]; then
        echo "❌ anchor 出现 $n 次: $A"; exit 1
    fi
done
n=$(grep -cF "kmer_vec" "$FILE" || true)
if [[ "$n" != "0" ]]; then echo "⚠️ 已应用过, 跳过"; exit 0; fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] 改动:"
    echo "  + uncertainty_weighted_contrastive 签名加 kmer_vec, jaccard_theta"
    echo "  + 在 soft_neg_mask 后追加 Jaccard mask 块"
    echo "  + forward 签名加 kmer_vec"
    exit 0
fi

cp "$FILE" "${FILE}.bak_v20A"
echo "📋 备份: ${FILE}.bak_v20A"

python - "$FILE" << 'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

# ── 1. uncertainty_weighted_contrastive 签名 ────────────────────
old_sig = '    def uncertainty_weighted_contrastive(self, pooled_emb, cluster_labels, u_epi, u_ale):'
new_sig = '    def uncertainty_weighted_contrastive(self, pooled_emb, cluster_labels, u_epi, u_ale, kmer_vec=None, jaccard_theta=0.18):'
assert src.count(old_sig) == 1
src = src.replace(old_sig, new_sig)

# ── 2. 在 "Queue 列屏蔽（新增）" 块后追加 Jaccard mask 块 ────────
# anchor: 找 n_soft_masked = ... 这一行 (in-batch + queue 软屏蔽完成后)
old_block = '''        n_soft_masked = int(high_sim_mask.sum().item()) + n_soft_masked_q

        # 分子：正样本 exp（不加权）'''
new_block = '''        n_soft_masked = int(high_sim_mask.sum().item()) + n_soft_masked_q

        # ── [v20A] Sequence-space Jaccard anchor mask ────────────────────────
        # 物理含义: cross-cluster + same-GT pair 的 5-mer Jaccard median ≈ 0.26
        #          cross-cluster + diff-GT pair 的 5-mer Jaccard median ≈ 0.14
        # 序列空间 Jaccard 不受 encoder drift 影响, 提供 invariant anchor.
        # 当两条 reads 异簇但 Jaccard > theta_j (推荐 0.18), 保护它们,
        # 不让 contrastive learning 把同源 fragments 推得更开.
        n_jaccard_masked = 0
        if kmer_vec is not None:
            with torch.no_grad():
                # In-batch Jaccard: |a & b| / |a | b|
                B_ = kmer_vec.shape[0]
                kv = kmer_vec.float()
                inter_b = kv @ kv.T                                              # (B, B)
                sums_b = kv.sum(dim=1, keepdim=True)                              # (B, 1)
                union_b = sums_b + sums_b.T - inter_b                             # (B, B)
                jacc_b = inter_b / union_b.clamp(min=1.0)                         # (B, B)
                anchor_mask_b = (jacc_b > jaccard_theta) & ~pos_mask_inbatch & ~self_mask
            neg_mask_full[:, :B] &= ~anchor_mask_b
            n_jaccard_masked = int(anchor_mask_b.sum().item())

            # Queue 列同样保护 (如果 model 维护了 queue_kmer)
            if sim_full.shape[1] > B and hasattr(self, 'queue_kmer'):
                q_count = sim_full.shape[1] - B
                with torch.no_grad():
                    q_kv = self.queue_kmer[:q_count].detach()                    # (Q, dim_k)
                    inter_q = kv @ q_kv.T                                         # (B, Q)
                    sums_q = q_kv.sum(dim=1, keepdim=True).T                      # (1, Q)
                    union_q = sums_b + sums_q - inter_q                           # (B, Q)
                    jacc_q = inter_q / union_q.clamp(min=1.0)
                    q_labels_l = self.queue_labels[:q_count].detach()
                    diff_label_q2 = (cluster_labels.unsqueeze(1) !=
                                     q_labels_l.unsqueeze(0))
                    anchor_mask_q = (jacc_q > jaccard_theta) & diff_label_q2
                neg_mask_full[:, B:] &= ~anchor_mask_q
                n_jaccard_masked += int(anchor_mask_q.sum().item())

        # 分子：正样本 exp（不加权）'''
assert src.count(old_block) == 1, f"v20A anchor block not unique: {src.count(old_block)}"
src = src.replace(old_block, new_block)

# ── 3. probe_stats 加 jaccard_masked 计数 ────────────────────────
old_probe = "        probe_stats['soft_neg_masked'] = n_soft_masked   # 软屏蔽计数（监控退火进度）"
new_probe = """        probe_stats['soft_neg_masked'] = n_soft_masked   # 软屏蔽计数（监控退火进度）
        probe_stats['jaccard_masked'] = n_jaccard_masked   # [v20A] Jaccard anchor 保护数"""
assert src.count(old_probe) == 1
src = src.replace(old_probe, new_probe)

# ── 4. forward 签名加 kmer_vec ────────────────────────────────────
old_fwd = '    def forward(self, reads, cluster_labels, consensus_target, epoch=0, round_idx=1):'
new_fwd = '    def forward(self, reads, cluster_labels, consensus_target, epoch=0, round_idx=1, kmer_vec=None, jaccard_theta=0.18):'
assert src.count(old_fwd) == 1
src = src.replace(old_fwd, new_fwd)

# ── 5. forward 内部把 kmer_vec 转发给 contrastive ──────────────
old_call = '''        con_loss, probe_stats = self.uncertainty_weighted_contrastive(
            pooled_emb, cluster_labels, u_epi, u_ale
        )'''
new_call = '''        con_loss, probe_stats = self.uncertainty_weighted_contrastive(
            pooled_emb, cluster_labels, u_epi, u_ale,
            kmer_vec=kmer_vec, jaccard_theta=jaccard_theta  # [v20A]
        )'''
assert src.count(old_call) == 1
src = src.replace(old_call, new_call)

# ── 6. outputs 加 jaccard_masked 输出 ────────────────────────────
old_out = "            'soft_neg_masked':  probe_stats.get('soft_neg_masked', 0),"
new_out = """            'soft_neg_masked':  probe_stats.get('soft_neg_masked', 0),
            'jaccard_masked':   probe_stats.get('jaccard_masked', 0),  # [v20A]"""
assert src.count(old_out) == 1
src = src.replace(old_out, new_out)

open(path, 'w').write(src)
print("✅ step1_model.py patched")
PYEOF

echo
echo "── grep 验证 ──"
grep -n "v20A\|kmer_vec\|jaccard_theta\|jaccard_masked" "$FILE" | head -15
echo
echo "✅ Patch 2/5 done"