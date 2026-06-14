#!/bin/bash
# =============================================================================
# v20.B Patch 2/4: step1_train.py
#   - 从 args 读 centroids_path (上轮 step2 存档)
#   - 加载 anchor_centroids 字典 (cluster_id → Tensor)
#   - model() 调用时传 anchor_centroids, lambda_topo
#   - 训练循环加 epoch_topo 累加和 epoch 末打印
# =============================================================================
set -e

FILE="${1:-/mnt/st_data/liangxinyi/code/models/step1_train.py}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$FILE" ]]; then echo "❌"; exit 1; fi

# ── anchor 检查 ──────────────────────────────────────────────────
ANCHOR_PREV='    prev_state   = getattr(args, '"'"'prev_state'"'"', None)'
ANCHOR_CALL='            loss_dict, outputs = model(reads_batch, labels_batch, consensus_batch, epoch=epoch, round_idx=round_idx)'
ANCHOR_HISTORY="        'contrastive_loss': [], 'reconstruction_loss': [], 'kl_loss': [],"
ANCHOR_PROBE='        if epoch_w_cc_cnt > 0 or epoch_w_da_cnt > 0:'

for A in "$ANCHOR_PREV" "$ANCHOR_CALL" "$ANCHOR_HISTORY" "$ANCHOR_PROBE"; do
    n=$(grep -cF "$A" "$FILE" || true)
    if [[ "$n" != "1" ]]; then
        echo "❌ anchor 出现 $n 次: $A"; exit 1
    fi
done
n=$(grep -cF "anchor_centroids\|lambda_topo\|epoch_topo" "$FILE" || true)
if [[ "$n" != "0" ]]; then echo "⚠️ 已应用过, 跳过"; exit 0; fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] 改动:"
    echo "  + 加载 anchor_centroids from centroids_path"
    echo "  + 训练循环传 anchor_centroids, lambda_topo 给 model"
    echo "  + epoch 末汇总 + 打印 topo loss"
    exit 0
fi

cp "$FILE" "${FILE}.bak_v20B"
echo "📋 备份: ${FILE}.bak_v20B"

python - "$FILE" << 'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

# ── 改动 1: 在 prev_state 后加载 anchor_centroids ──────────────────
old1 = "    prev_state   = getattr(args, 'prev_state', None)"
new1 = """    prev_state   = getattr(args, 'prev_state', None)

    # [v20B] Topology preservation: 加载上轮 step2 存档的 centroids 作为 anchor
    centroids_path = getattr(args, 'centroids_path', None)
    lambda_topo = float(getattr(args, 'lambda_topo', 0.0))
    anchor_centroids = None
    if centroids_path is not None and lambda_topo > 0 and round_idx >= 2:
        if os.path.exists(centroids_path):
            try:
                _centroid_data = torch.load(centroids_path, map_location='cpu')
                # 兼容: 可能是 dict {cid: tensor} 也可能 wrapped {'centroids': dict}
                if isinstance(_centroid_data, dict) and 'centroids' in _centroid_data:
                    anchor_centroids = _centroid_data['centroids']
                else:
                    anchor_centroids = _centroid_data
                if isinstance(anchor_centroids, dict):
                    print(f"   🔗 [v20B] anchor_centroids: {len(anchor_centroids)} clusters "
                          f"(λ_topo={lambda_topo})")
                else:
                    print(f"   ⚠️ [v20B] centroids_path 格式不对, 跳过 topology")
                    anchor_centroids = None
            except Exception as e:
                print(f"   ⚠️ [v20B] centroids 加载失败 ({e}), 跳过 topology")
                anchor_centroids = None
        else:
            print(f"   ⚠️ [v20B] centroids_path 不存在: {centroids_path}")"""
assert src.count(old1) == 1
src = src.replace(old1, new1)

# ── 改动 2: training_history 加 topo_loss ──────────────────────────
old2 = "        'contrastive_loss': [], 'reconstruction_loss': [], 'kl_loss': [],"
new2 = "        'contrastive_loss': [], 'reconstruction_loss': [], 'kl_loss': [], 'topo_loss': [],  # [v20B]"
assert src.count(old2) == 1
src = src.replace(old2, new2)

# ── 改动 3: 训练循环传 anchor + 累加 epoch_topo ────────────────────
# 先在 epoch 累加变量初始化处加 epoch_topo
old3 = """            consensus_batch  = batch_data['consensus_target'].to(device)

            # [FIX-P0] 传入 consensus_target
            loss_dict, outputs = model(reads_batch, labels_batch, consensus_batch, epoch=epoch, round_idx=round_idx)"""
new3 = """            consensus_batch  = batch_data['consensus_target'].to(device)

            # [FIX-P0] 传入 consensus_target | [v20B] 传入 anchor_centroids, lambda_topo
            loss_dict, outputs = model(
                reads_batch, labels_batch, consensus_batch,
                epoch=epoch, round_idx=round_idx,
                anchor_centroids=anchor_centroids, lambda_topo=lambda_topo,
            )"""
assert src.count(old3) == 1
src = src.replace(old3, new3)

# ── 改动 4: epoch_topo 累加 (在 epoch_con += ... 附近找单一 anchor) ─
# 我们用 'epoch_con   += loss_dict' 作为锚 (从前面 grep 看是唯一)
old4 = "            epoch_con   += loss_dict['contrastive'].item()"
new4 = """            epoch_con   += loss_dict['contrastive'].item()
            epoch_topo  += float(outputs.get('topo_loss', 0.0))            # [v20B]
            epoch_n_topo_used += int(outputs.get('n_topo_used', 0))         # [v20B]"""
assert src.count(old4) == 1
src = src.replace(old4, new4)

# ── 改动 5: epoch_topo 初始化 (跟 epoch_con = 0 在同一处) ──────────
# 找 epoch summary 累加器初始化:
old5 = """        training_history['contrastive_loss'].append(avg(epoch_con))"""
new5 = """        training_history['contrastive_loss'].append(avg(epoch_con))
        training_history['topo_loss'].append(avg(epoch_topo))           # [v20B]"""
assert src.count(old5) == 1
src = src.replace(old5, new5)

# 6. 在 epoch 循环开头初始化 epoch_topo (找 chained assignment 行)
old6 = "        epoch_w_cc_cnt = epoch_w_da_cnt = epoch_cos_pos_cnt = epoch_cos_neg_cnt = 0"
new6 = """        epoch_w_cc_cnt = epoch_w_da_cnt = epoch_cos_pos_cnt = epoch_cos_neg_cnt = 0
        epoch_topo = 0          # [v20B] topology regularizer running sum
        epoch_n_topo_used = 0   # [v20B] count of clusters constrained per epoch"""
assert src.count(old6) == 1, f"epoch_init anchor 不唯一: {src.count(old6)}"
src = src.replace(old6, new6)

# ── 改动 7: epoch 末打印 topo (在探针 B 前) ────────────────────────
old7 = "        if epoch_w_cc_cnt > 0 or epoch_w_da_cnt > 0:"
new7 = """        # [v20B] Topology preservation 监控
        if anchor_centroids is not None and lambda_topo > 0 and round_idx >= 2:
            avg_topo = epoch_topo / max(len(loader), 1)
            avg_used = epoch_n_topo_used / max(len(loader), 1)
            print(f"   🌐 [v20B] topo_loss avg: {avg_topo:.6f}  "
                  f"(λ={lambda_topo}, ~{avg_used:.0f} clusters/batch)")

        if epoch_w_cc_cnt > 0 or epoch_w_da_cnt > 0:"""
assert src.count(old7) == 1
src = src.replace(old7, new7)

open(path, 'w').write(src)
print("✅ step1_train.py patched")
PYEOF

echo
echo "── grep 验证 ──"
grep -n "v20B\|anchor_centroids\|lambda_topo\|epoch_topo\|topo_loss" "$FILE" | head -20
echo
echo "✅ Patch 2/4 done"