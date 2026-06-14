#!/usr/bin/env bash
# patch_step2args_split.sh
# ========================
# 修复拆分空转的最后一棒:
# main_loop.py 用 argparse.Namespace 手动重建 step2_args 传给 run_step2,
# 但漏了 enable_split / split_tau / split_min_size 三个字段, 导致
# step2_runner 里 getattr(args,'enable_split',False) 永远取到 False -> 拆分不触发。
# 本补丁在 step2_args Namespace 里补上这三个字段。
#
# 安全: 备份 .bak_step2args / 幂等 / 默认 dry-run。
#
# 用法(在 code 根 或 models/ 下都行):
#   bash patch_step2args_split.sh           # dry-run
#   bash patch_step2args_split.sh --apply
#   bash patch_step2args_split.sh --revert

set -euo pipefail

TARGET=""
for cand in "models/main_loop.py" "main_loop.py" "../models/main_loop.py"; do
  if [[ -f "$cand" ]]; then TARGET="$cand"; break; fi
done
if [[ -z "$TARGET" ]]; then
  echo "❌ 找不到 main_loop.py(试过 models/ , ./ , ../models/)。"; exit 1
fi
echo "[target] $TARGET"
BAK="${TARGET}.bak_step2args"
MARKER="# [v21-SPLIT-STEP2ARGS]"

MODE="dryrun"
[[ "${1:-}" == "--apply" ]] && MODE="apply"
[[ "${1:-}" == "--revert" ]] && MODE="revert"

if [[ "$MODE" == "revert" ]]; then
  if [[ -f "$BAK" ]]; then cp "$BAK" "$TARGET"; echo "✅ 已还原 $TARGET"; else echo "❌ 无备份 $BAK"; exit 1; fi
  exit 0
fi

if grep -qF "$MARKER" "$TARGET"; then
  echo "ℹ️  已含 step2_args 拆分字段, 跳过。重打先 --revert"; exit 0
fi

# 锚点: step2_args 里的 rebirth_mode 行(其后紧跟 ')')
ANCHOR="            rebirth_mode=getattr(args, 'rebirth_mode', 'nearest'),"
if ! grep -qF "$ANCHOR" "$TARGET"; then
  echo "❌ 找不到锚点行(step2_args 的 rebirth_mode=...)。"
  echo "   请确认 main_loop.py 的 step2_args Namespace 结构未变。"
  exit 1
fi

echo "将在 step2_args 的 rebirth_mode 字段后插入三个拆分字段。"
if [[ "$MODE" == "dryrun" ]]; then
  echo "🔍 DRY-RUN, 未改动。确认后: bash $0 --apply"; exit 0
fi

cp "$TARGET" "$BAK"; echo "💾 备份: $BAK"

python3 - "$TARGET" <<'PYEOF'
import sys
target = sys.argv[1]
anchor = "            rebirth_mode=getattr(args, 'rebirth_mode', 'nearest'),"
inject = (
    anchor + "\n"
    "            # [v21-SPLIT-STEP2ARGS] 透传拆分开关到 run_step2\n"
    "            enable_split=getattr(args, 'enable_split', False),\n"
    "            split_tau=getattr(args, 'split_tau', 5),\n"
    "            split_min_size=getattr(args, 'split_min_size', 6),"
)
with open(target, encoding='utf-8') as f:
    src = f.read()
if anchor not in src:
    print("❌ 锚点丢失"); sys.exit(1)
# 只替换第一处(step2_args 内), 避免误伤其它 rebirth_mode 出现
src = src.replace(anchor, inject, 1)
with open(target, 'w', encoding='utf-8') as f:
    f.write(src)
print("✅ 注入完成")
PYEOF

python3 -m py_compile "$TARGET" && echo "   ✓ main_loop.py 语法OK" || { echo "❌ 语法错误, 还原"; cp "$BAK" "$TARGET"; exit 1; }

echo ""
echo "✅ 最后一棒接通。step2_args 现已携带 enable_split/split_tau/split_min_size。"
echo "   直接重跑原三轮命令(已带 --enable_split --split_tau 5 --split_min_size 6)即可。"
echo "   验证: 重跑日志应出现 '✂️  [v21] 簇内拆分' 块, 且簇数 11648 → 12000+。"