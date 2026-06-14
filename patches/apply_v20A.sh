#!/bin/bash
# =============================================================================
# v20.A 主控脚本
# 用法:
#   ./apply_v20A.sh dry-run              # 全部 patch dry-run
#   ./apply_v20A.sh apply                # 实际应用
#   ./apply_v20A.sh rollback             # 回滚到 .bak_v20A
#   ./apply_v20A.sh status               # 查看应用状态
# =============================================================================
set -e

CODE_DIR="${CODE_DIR:-/mnt/st_data/liangxinyi/code}"
PATCH_DIR="$(dirname "$(realpath "$0")")"
F1="$CODE_DIR/models/step1_data.py"
F2="$CODE_DIR/models/step1_model.py"
F3="$CODE_DIR/models/step1_train.py"
F4="$CODE_DIR/models/main_loop.py"

cmd="${1:-status}"

case "$cmd" in
  dry-run)
    echo "═══ v20.A Dry Run ═══"
    DRY_RUN=1 bash "$PATCH_DIR/v20A_patch1_data.sh" "$F1"
    DRY_RUN=1 bash "$PATCH_DIR/v20A_patch2_model.sh" "$F2"
    DRY_RUN=1 bash "$PATCH_DIR/v20A_patch3_train.sh" "$F3"
    DRY_RUN=1 bash "$PATCH_DIR/v20A_patch4_main.sh" "$F4"
    echo
    echo "✅ Dry-run 全部通过. 用 'apply' 真实应用."
    ;;
  apply)
    echo "═══ v20.A Apply ═══"
    bash "$PATCH_DIR/v20A_patch1_data.sh" "$F1"
    echo
    bash "$PATCH_DIR/v20A_patch2_model.sh" "$F2"
    echo
    bash "$PATCH_DIR/v20A_patch3_train.sh" "$F3"
    echo
    bash "$PATCH_DIR/v20A_patch4_main.sh" "$F4"
    echo
    echo "✅ v20.A 全部应用. 备份在 .bak_v20A"
    echo "回滚: $0 rollback"
    ;;
  rollback)
    echo "═══ v20.A Rollback ═══"
    for F in "$F1" "$F2" "$F3" "$F4"; do
        if [[ -f "${F}.bak_v20A" ]]; then
            cp "${F}.bak_v20A" "$F"
            echo "↩️  恢复: $F"
        else
            echo "⚠️  无备份: ${F}.bak_v20A"
        fi
    done
    ;;
  status)
    echo "═══ v20.A Status ═══"
    for F in "$F1" "$F2" "$F3" "$F4"; do
        if grep -q "v20A\|jaccard_theta\|kmer_vec\|seq_to_kmer_bitvec" "$F" 2>/dev/null; then
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