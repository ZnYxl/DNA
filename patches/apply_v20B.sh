#!/bin/bash
# =============================================================================
# v20.B 主控脚本
# 用法:
#   ./apply_v20B.sh dry-run
#   ./apply_v20B.sh apply
#   ./apply_v20B.sh rollback
#   ./apply_v20B.sh status
# =============================================================================
set -e

CODE_DIR="${CODE_DIR:-/mnt/st_data/liangxinyi/code}"
PATCH_DIR="$(dirname "$(realpath "$0")")"
F1="$CODE_DIR/models/step1_model.py"
F2="$CODE_DIR/models/step1_train.py"
F3="$CODE_DIR/models/main_loop.py"

cmd="${1:-status}"

case "$cmd" in
  dry-run)
    echo "═══ v20.B Dry Run ═══"
    DRY_RUN=1 bash "$PATCH_DIR/v20B_patch1_model.sh" "$F1"
    DRY_RUN=1 bash "$PATCH_DIR/v20B_patch2_train.sh" "$F2"
    DRY_RUN=1 bash "$PATCH_DIR/v20B_patch3_main.sh"  "$F3"
    echo
    echo "✅ Dry-run 全部通过."
    ;;
  apply)
    echo "═══ v20.B Apply ═══"
    bash "$PATCH_DIR/v20B_patch1_model.sh" "$F1"
    echo
    bash "$PATCH_DIR/v20B_patch2_train.sh" "$F2"
    echo
    bash "$PATCH_DIR/v20B_patch3_main.sh"  "$F3"
    echo
    echo "✅ v20.B 全部应用. 备份在 .bak_v20B"
    ;;
  rollback)
    echo "═══ v20.B Rollback ═══"
    for F in "$F1" "$F2" "$F3"; do
        if [[ -f "${F}.bak_v20B" ]]; then
            cp "${F}.bak_v20B" "$F"
            echo "↩️  恢复: $F"
        else
            echo "⚠️  无备份: ${F}.bak_v20B"
        fi
    done
    ;;
  status)
    echo "═══ v20.B Status ═══"
    for F in "$F1" "$F2" "$F3"; do
        if grep -q "v20B\|topology_regularizer\|lambda_topo\|anchor_centroids" "$F" 2>/dev/null; then
            echo "✅ APPLIED: $F"
        else
            echo "⬜ CLEAN:    $F"
        fi
    done
    ;;
  *)
    echo "用法: $0 {dry-run|apply|rollback|status}"
    exit 1
    ;;
esac