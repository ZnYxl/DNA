#!/usr/bin/env bash
# =============================================================================
# patch_clean_mode_zone3_log.sh
# 软大扫除 · 第三刀 (Zone III 隔离 - 仅修正误导日志)
# -----------------------------------------------------------------------------
# 背景: Zone III 真正的隔离动作 (new_labels[zone_ids==3] = -1) 早已被
#       [DISABLE-ZONE3] 注释成 pass, 所有轮次都不隔离 (终态设计)。
#       但 493 行 print 仍打印 "Zone III 标签隔离: N reads → -1", 与实际
#       行为不符 —— 导师看日志会误以为 Zone III read 被丢成 -1。
#
# 作用: 仅把这句 print 的文字改成实话 (隔离已禁用, read 保留原簇标签)。
#       不动任何逻辑、不碰 z3_count/z3_indices/z3_original_labels 变量。
#       零逻辑变更 -> 不可能影响 SR, apply 后无需重跑。
#
# 特性: .bak 备份 + grep 幂等 + --dry-run 预览
# 用法:
#   bash patch_clean_mode_zone3_log.sh --dry-run
#   bash patch_clean_mode_zone3_log.sh
#   bash patch_clean_mode_zone3_log.sh --revert
# =============================================================================
set -euo pipefail

TARGET="/mnt/st_data/liangxinyi/code/models/step2_runner.py"
BAK="${TARGET}.bak_cleanmode3"
MARK="v22-CLEANMODE-3"

DRY=0; REVERT=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --revert)  REVERT=1 ;;
    *) echo "未知参数: $a"; exit 1 ;;
  esac
done

[ -f "$TARGET" ] || { echo "❌ 找不到 $TARGET"; exit 1; }

if [ "$REVERT" -eq 1 ]; then
  [ -f "$BAK" ] || { echo "❌ 无备份 $BAK, 无法回滚"; exit 1; }
  cp "$BAK" "$TARGET"
  echo "✅ 已从 $BAK 回滚 $TARGET"
  exit 0
fi

if grep -qF "$MARK" "$TARGET"; then
  echo "⏭️  已打过本补丁 ($MARK 存在), 跳过。如需重打请先 --revert。"
  exit 0
fi

# ---- 精确定位 old_str (逐字符匹配) ----
OLD='    print(f"   🔒 Zone III 标签隔离: {z3_count} reads → -1 (保留原始标签供 consensus 软参与)")'
NEW='    print(f"   🔓 Zone III 隔离已禁用 (终态): {z3_count} reads 保留原簇标签, 不隔离为 -1")  # ['"$MARK"']'

c=$(grep -F -c "$OLD" "$TARGET" || true)
echo "🔎 old_str 匹配次数: $c (期望 1)"
if [ "$c" -ne 1 ]; then
  echo "❌ old_str 不是恰好唯一匹配。服务器代码可能已变动, 中止以免误改。"
  echo "   请贴回: grep -n 'Zone III 标签隔离' '$TARGET'"
  exit 1
fi

TMP="$(mktemp)"
cp "$TARGET" "$TMP"
python3 - "$TMP" "$OLD" "$NEW" <<'PY'
import sys
path, old, new = sys.argv[1:4]
s = open(path, encoding='utf-8').read()
assert s.count(old) == 1, "唯一性二次校验失败"
s = s.replace(old, new)
open(path, 'w', encoding='utf-8').write(s)
PY

echo ""
echo "===== DIFF 预览 (- 原 / + 改后) ====="
diff -u "$TARGET" "$TMP" || true
echo "====================================="

if [ "$DRY" -eq 1 ]; then
  rm -f "$TMP"
  echo ""
  echo "🧪 dry-run 完成, 原文件未改动。确认只改了这一行 print 文字, 去掉 --dry-run 再跑。"
  exit 0
fi

cp "$TARGET" "$BAK"
echo "💾 已备份: $BAK"
cp "$TMP" "$TARGET"
rm -f "$TMP"

if grep -qF "$MARK" "$TARGET"; then
  echo "✅ 补丁已应用 (仅日志文字, 零逻辑变更)。"
  grep -n "$MARK" "$TARGET" || true
else
  echo "⚠️ 应用后未检出标记, 请人工核对。"
fi

echo ""
echo "说明: 本刀仅改 print 字符串, 不影响任何计算 -> 无需重跑验证。"
echo "回滚: bash $(basename "$0") --revert"