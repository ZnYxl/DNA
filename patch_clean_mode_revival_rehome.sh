#!/usr/bin/env bash
# =============================================================================
# patch_clean_mode_revival_rehome.sh
# 软大扫除 · 第一刀 (最安全的两个空转机制)
# -----------------------------------------------------------------------------
# 作用: 给 step2_runner.py 的两段空转逻辑各加一个 clean_mode 短路:
#   1. 死数据复活    (if len(noise_indices) > 0 ...)
#   2. Zone III 归巢 (if len(final_noise_indices) > 0 ...)
# clean_mode=True 时条件短路为 False -> 走原有 else "跳过"分支。
# 三轮实测本就走 else, 故 clean_mode 关闭时行为逐字节不变, SR 守 0.9699。
#
# 不碰: eval_labels 定义(540) / final_noise_mask 定义(541) / 任何下游变量。
#
# 特性: .bak 备份 + grep 幂等(已打过补丁则跳过) + --dry-run 预览
# 用法:
#   bash patch_clean_mode_revival_rehome.sh --dry-run   # 先看 diff
#   bash patch_clean_mode_revival_rehome.sh             # 实际应用
#   bash patch_clean_mode_revival_rehome.sh --revert    # 从 .bak 回滚
# =============================================================================
set -euo pipefail

TARGET="/mnt/st_data/liangxinyi/code/models/step2_runner.py"
BAK="${TARGET}.bak_cleanmode1"
MARK="[v22-CLEANMODE-1]"

DRY=0; REVERT=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --revert)  REVERT=1 ;;
    *) echo "未知参数: $a"; exit 1 ;;
  esac
done

[ -f "$TARGET" ] || { echo "❌ 找不到 $TARGET"; exit 1; }

# ---- revert ----
if [ "$REVERT" -eq 1 ]; then
  [ -f "$BAK" ] || { echo "❌ 无备份 $BAK, 无法回滚"; exit 1; }
  cp "$BAK" "$TARGET"
  echo "✅ 已从 $BAK 回滚 $TARGET"
  exit 0
fi

# ---- 幂等检查 ----
if grep -q "$MARK" "$TARGET"; then
  echo "⏭️  已打过本补丁 ($MARK 存在), 跳过。如需重打请先 --revert。"
  exit 0
fi

# ---- 精确定位两处 old_str (逐字符匹配, 不靠行号) ----
OLD1='    if len(noise_indices) > 0 and len(centroids) > 0:'
NEW1="    if (not getattr(args, 'clean_mode', False)) and len(noise_indices) > 0 and len(centroids) > 0:  # ${MARK} 死数据复活"

OLD2='    if len(final_noise_indices) > 0 and len(centroids) > 0:'
NEW2="    if (not getattr(args, 'clean_mode', False)) and len(final_noise_indices) > 0 and len(centroids) > 0:  # ${MARK} Zone III 归巢"

# ---- 唯一性校验 (每个 old_str 必须恰好出现 1 次) ----
c1=$(grep -F -c "$OLD1" "$TARGET" || true)
c2=$(grep -F -c "$OLD2" "$TARGET" || true)
echo "🔎 匹配次数: 死数据复活 if = $c1 | 归巢 if = $c2"
if [ "$c1" -ne 1 ] || [ "$c2" -ne 1 ]; then
  echo "❌ old_str 不是恰好唯一匹配 (期望各 1 次)。"
  echo "   服务器代码可能已变动, 中止以免误改。请把以下两行的真实原文贴回:"
  echo "   grep -n 'len(noise_indices) > 0' '$TARGET'"
  echo "   grep -n 'len(final_noise_indices) > 0' '$TARGET'"
  exit 1
fi

# ---- dry-run: 用临时文件预览 diff, 不改原文件 ----
TMP="$(mktemp)"
cp "$TARGET" "$TMP"
python3 - "$TMP" "$OLD1" "$NEW1" "$OLD2" "$NEW2" <<'PY'
import sys
path, o1, n1, o2, n2 = sys.argv[1:6]
s = open(path, encoding='utf-8').read()
assert s.count(o1) == 1 and s.count(o2) == 1, "唯一性二次校验失败"
s = s.replace(o1, n1).replace(o2, n2)
open(path, 'w', encoding='utf-8').write(s)
PY

echo ""
echo "===== DIFF 预览 (- 原 / + 改后) ====="
diff -u "$TARGET" "$TMP" || true
echo "====================================="

if [ "$DRY" -eq 1 ]; then
  rm -f "$TMP"
  echo ""
  echo "🧪 dry-run 完成, 原文件未改动。确认 diff 只动了那两行 if 后, 去掉 --dry-run 再跑。"
  exit 0
fi

# ---- 实际应用: 先备份, 再落盘 ----
cp "$TARGET" "$BAK"
echo "💾 已备份: $BAK"
cp "$TMP" "$TARGET"
rm -f "$TMP"

# ---- 应用后复核 ----
if grep -q "$MARK" "$TARGET" && [ "$(grep -F -c "$MARK 死数据复活" "$TARGET")" -ge 0 ]; then
  echo "✅ 补丁已应用。"
  echo "   验证: grep -n '$MARK' '$TARGET'"
  grep -n "$MARK" "$TARGET" || true
else
  echo "⚠️ 应用后未检出标记, 请人工核对。"
fi

echo ""
echo "下一步:"
echo "  1. 不开 clean_mode 重跑三轮 -> SR 必须仍是 0.9699 (clean_mode 默认 False, 行为不变)"
echo "  2. 守住后, 等 main_loop 加 --clean_mode 透传, 再开 clean_mode 重跑 -> SR 仍 0.9699 = 证明两机制空转"
echo "  回滚: bash $(basename "$0") --revert"