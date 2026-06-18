#!/usr/bin/env bash
# =============================================================================
# patch_clean_mode_rebirth.sh
# 软大扫除 · 第二刀 (Rebirth)
# -----------------------------------------------------------------------------
# 作用: clean_mode=True 时强制 _rebirth_mode='off', 复用 Rebirth 自带的
#       "已禁用 (--rebirth_mode off)" 分支, 不新增 if 包整段。
#
# 已验证: --rebirth_mode off 三轮 SR = nearest 三轮 SR (0.9539/0.9662/0.9699,
#         簇数 15663/16761/17076 完全一致) -> Rebirth 在 Seq_1D 上空转, 零贡献。
#
# 改动: 在 _rebirth_mode = getattr(...) 那行之后插入 2 行 clean_mode 短路。
# 不碰: reborn_mask / zone_ids / new_labels / 任何下游变量。
# clean_mode=False(默认)时, _rebirth_mode 仍取 args 原值, 行为逐字节不变。
#
# 特性: .bak 备份 + grep 幂等 + --dry-run 预览
# 用法:
#   bash patch_clean_mode_rebirth.sh --dry-run
#   bash patch_clean_mode_rebirth.sh
#   bash patch_clean_mode_rebirth.sh --revert
# =============================================================================
set -euo pipefail

TARGET="/mnt/st_data/liangxinyi/code/models/step2_runner.py"
BAK="${TARGET}.bak_cleanmode2"
MARK="v22-CLEANMODE-2"

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
if grep -qF "$MARK" "$TARGET"; then
  echo "⏭️  已打过本补丁 ($MARK 存在), 跳过。如需重打请先 --revert。"
  exit 0
fi

# ---- 精确定位 anchor (逐字符匹配, 不靠行号) ----
ANCHOR="    _rebirth_mode = getattr(args, 'rebirth_mode', 'nearest')"
# 插入内容: 在 anchor 行之后追加 2 行 (保持 4 空格缩进)
INSERT="    if getattr(args, 'clean_mode', False):  # [${MARK}] clean_mode 下强制关闭 Rebirth (复用 off 分支)
        _rebirth_mode = 'off'"

# ---- 唯一性校验 ----
c=$(grep -F -c "$ANCHOR" "$TARGET" || true)
echo "🔎 anchor 匹配次数: $c (期望 1)"
if [ "$c" -ne 1 ]; then
  echo "❌ anchor 不是恰好唯一匹配。服务器代码可能已变动, 中止以免误改。"
  echo "   请贴回: grep -n \"_rebirth_mode = getattr\" '$TARGET'"
  exit 1
fi

# ---- dry-run: 临时文件预览 diff ----
TMP="$(mktemp)"
cp "$TARGET" "$TMP"
python3 - "$TMP" "$ANCHOR" "$INSERT" <<'PY'
import sys
path, anchor, insert = sys.argv[1:4]
lines = open(path, encoding='utf-8').read().split('\n')
out, hit = [], 0
for ln in lines:
    out.append(ln)
    if ln == anchor:
        out.extend(insert.split('\n'))
        hit += 1
assert hit == 1, f"anchor 命中 {hit} 次, 期望 1"
open(path, 'w', encoding='utf-8').write('\n'.join(out))
PY

echo ""
echo "===== DIFF 预览 (- 原 / + 改后) ====="
diff -u "$TARGET" "$TMP" || true
echo "====================================="

if [ "$DRY" -eq 1 ]; then
  rm -f "$TMP"
  echo ""
  echo "🧪 dry-run 完成, 原文件未改动。确认 diff 只在 _rebirth_mode 行后插了 2 行, 去掉 --dry-run 再跑。"
  exit 0
fi

# ---- 实际应用 ----
cp "$TARGET" "$BAK"
echo "💾 已备份: $BAK"
cp "$TMP" "$TARGET"
rm -f "$TMP"

if grep -qF "$MARK" "$TARGET"; then
  echo "✅ 补丁已应用。"
  grep -n "$MARK" "$TARGET" || true
else
  echo "⚠️ 应用后未检出标记, 请人工核对。"
fi

echo ""
echo "下一步:"
echo "  此补丁只在 clean_mode=True 时生效。clean_mode 默认 False, 行为不变。"
echo "  真正验证留到 main_loop 加 --clean_mode 透传后, 开 clean_mode 统一重跑。"
echo "  回滚: bash $(basename "$0") --revert"