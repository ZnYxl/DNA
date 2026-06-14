#!/bin/bash
# =============================================================================
# v20.A Patch 4/5: main_loop.py
#   - 加 --jaccard_k 和 --jaccard_theta CLI 参数
# =============================================================================
set -e

FILE="${1:-main_loop.py}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$FILE" ]]; then echo "❌"; exit 1; fi

ANCHOR="    parser.add_argument('--consensus_source', type=str, default='mv',"
n=$(grep -cF "$ANCHOR" "$FILE" || true)
if [[ "$n" != "1" ]]; then echo "❌ anchor 出现 $n 次"; exit 1; fi
n=$(grep -cF "jaccard_theta" "$FILE" || true)
if [[ "$n" != "0" ]]; then echo "⚠️ 已应用过, 跳过"; exit 0; fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] 改动: 加 --jaccard_k, --jaccard_theta CLI"
    exit 0
fi

cp "$FILE" "${FILE}.bak_v20A"
echo "📋 备份: ${FILE}.bak_v20A"

python - "$FILE" << 'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()

old = "    parser.add_argument('--consensus_source', type=str, default='mv',"
new = """    # [v20A] Jaccard mask CLI parameters
    parser.add_argument('--jaccard_k', type=int, default=5,
                        help='[v20A] k-mer size for sequence-space Jaccard mask')
    parser.add_argument('--jaccard_theta', type=float, default=0.18,
                        help='[v20A] Jaccard threshold for cross-cluster anchor protection. '
                             'Set to 1.01 to disable. Default 0.18 from spike (k=5).')

    parser.add_argument('--consensus_source', type=str, default='mv',"""
assert src.count(old) == 1
src = src.replace(old, new)

open(path, 'w').write(src)
print("✅ main_loop.py patched")
PYEOF

echo
echo "── grep ──"
grep -n "jaccard" "$FILE" | head -10
echo
echo "✅ Patch 4/5 done"