#!/bin/bash
# =============================================================================
# v20.A Patch 3/5: step1_train.py
#   - 训练 batch loop 取 batch_data['kmer_vec'] 送 GPU
#   - 调用 model() 时传 kmer_vec, jaccard_theta
# =============================================================================
set -e

FILE="${1:-models/step1_train.py}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$FILE" ]]; then echo "❌"; exit 1; fi

ANCHOR_BATCH="            consensus_batch  = batch_data['consensus_target'].to(device)"
ANCHOR_CALL="            loss_dict, outputs = model(reads_batch, labels_batch, consensus_batch, epoch=epoch, round_idx=round_idx)"

n=$(grep -cF "$ANCHOR_BATCH" "$FILE" || true)
if [[ "$n" != "1" ]]; then echo "❌ batch anchor 出现 $n 次"; exit 1; fi
n=$(grep -cF "$ANCHOR_CALL" "$FILE" || true)
if [[ "$n" != "1" ]]; then echo "❌ call anchor 出现 $n 次"; exit 1; fi
n=$(grep -cF "kmer_vec" "$FILE" || true)
if [[ "$n" != "0" ]]; then echo "⚠️ 已应用过, 跳过"; exit 0; fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] 改动: 训练循环加 kmer_batch + 转发"
    exit 0
fi

cp "$FILE" "${FILE}.bak_v20A"
echo "📋 备份: ${FILE}.bak_v20A"

python - "$FILE" << 'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

# ── 改 Step1Dataset 创建 (传 kmer_k) ──────────────────────────
old_ds = '''        max_reads_per_cluster=getattr(args, 'max_reads_per_cluster', 30),  # 与 FedDNA 对齐，每簇最多30条
    )'''
new_ds = '''        max_reads_per_cluster=getattr(args, 'max_reads_per_cluster', 30),  # 与 FedDNA 对齐，每簇最多30条
        kmer_k=int(getattr(args, 'jaccard_k', 5)),  # [v20A] sequence-space Jaccard k-mer
    )'''
assert src.count(old_ds) == 1, f"Step1Dataset anchor 不唯一: {src.count(old_ds)}"
src = src.replace(old_ds, new_ds)

# ── 改训练循环 ────────────────────────────────────────────────
old = '''            consensus_batch  = batch_data['consensus_target'].to(device)

            # [FIX-P0] 传入 consensus_target
            loss_dict, outputs = model(reads_batch, labels_batch, consensus_batch, epoch=epoch, round_idx=round_idx)'''
new = '''            consensus_batch  = batch_data['consensus_target'].to(device)
            # [v20A] sequence-space k-mer bitvec for Jaccard mask
            kmer_batch = batch_data.get('kmer_vec')
            if kmer_batch is not None:
                kmer_batch = kmer_batch.to(device)
            jaccard_theta = float(getattr(args, 'jaccard_theta', 0.18))

            # [FIX-P0] 传入 consensus_target | [v20A] 传入 kmer_vec, jaccard_theta
            loss_dict, outputs = model(
                reads_batch, labels_batch, consensus_batch,
                epoch=epoch, round_idx=round_idx,
                kmer_vec=kmer_batch, jaccard_theta=jaccard_theta
            )'''
assert src.count(old) == 1
src = src.replace(old, new)

open(path, 'w').write(src)
print("✅ step1_train.py patched")
PYEOF

echo
echo "── grep ──"
grep -n "kmer\|jaccard" "$FILE" | head -10
echo
echo "✅ Patch 3/5 done"