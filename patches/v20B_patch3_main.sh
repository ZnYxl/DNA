#!/bin/bash
# =============================================================================
# v20.B Patch 3/4: main_loop.py
#   - 加 --lambda_topo CLI 参数 (默认 0.1)
#   - step1_args 传 centroids_path=current_centroids_path, lambda_topo
# =============================================================================
set -e

FILE="${1:-/mnt/st_data/liangxinyi/code/models/main_loop.py}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$FILE" ]]; then echo "❌"; exit 1; fi

ANCHOR_CLI="    parser.add_argument('--rebirth_mode', type=str, default='nearest',"
ANCHOR_STEP1="                max_reads_per_cluster=getattr(args, 'max_reads_per_cluster', 30),"

for A in "$ANCHOR_CLI" "$ANCHOR_STEP1"; do
    n=$(grep -cF "$A" "$FILE" || true)
    if [[ "$n" != "1" ]]; then echo "❌ anchor 出现 $n 次: $A"; exit 1; fi
done
n=$(grep -cF "lambda_topo" "$FILE" || true)
if [[ "$n" != "0" ]]; then echo "⚠️ 已应用过, 跳过"; exit 0; fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] 改动:"
    echo "  + --lambda_topo CLI"
    echo "  + step1_args 加 centroids_path, lambda_topo"
    exit 0
fi

cp "$FILE" "${FILE}.bak_v20B"
echo "📋 备份: ${FILE}.bak_v20B"

python - "$FILE" << 'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

# ── 改动 1: 加 --lambda_topo CLI ───────────────────────────────────
old_cli = "    parser.add_argument('--rebirth_mode', type=str, default='nearest',"
new_cli = """    # [v20B] Topology preservation regularizer
    parser.add_argument('--lambda_topo', type=float, default=0.1,
                        help='[v20B] Cluster topology preservation weight. '
                             'L_topo = (1 - cos(c_now, c_anchor)) * cluster_size. '
                             'Set to 0 to disable. R≥2 only. Default 0.1 (gentle pull).')

    parser.add_argument('--rebirth_mode', type=str, default='nearest',"""
assert src.count(old_cli) == 1
src = src.replace(old_cli, new_cli)

# ── 改动 2: step1_args 加 centroids_path 和 lambda_topo ────────────
old_args = "                max_reads_per_cluster=getattr(args, 'max_reads_per_cluster', 30),"
new_args = """                max_reads_per_cluster=getattr(args, 'max_reads_per_cluster', 30),
                centroids_path=current_centroids_path,                  # [v20B]
                lambda_topo=getattr(args, 'lambda_topo', 0.1),           # [v20B]"""
assert src.count(old_args) == 1
src = src.replace(old_args, new_args)

open(path, 'w').write(src)
print("✅ main_loop.py patched")
PYEOF

echo
echo "── grep ──"
grep -n "v20B\|lambda_topo\|centroids_path" "$FILE" | head -10
echo
echo "✅ Patch 3/4 done"