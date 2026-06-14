#!/usr/bin/env bash
# patch_main_loop_split_args.sh
# =============================
# 在 main_loop.py 的 argparse 中加入 v21 拆分开关:
#   --enable_split / --split_tau / --split_min_size
# 这三个参数随 args 透传到 run_step2(getattr 读取)。
#
# 安全: 备份 .bak_split / 幂等检查 / 默认 dry-run。
#
# 用法:
#   bash patch_main_loop_split_args.sh           # dry-run
#   bash patch_main_loop_split_args.sh --apply
#   bash patch_main_loop_split_args.sh --revert

set -euo pipefail

# 自动定位 main_loop.py(根 / models 子目录 / 当前目录 均可)
TARGET=""
for cand in "models/main_loop.py" "main_loop.py" "../models/main_loop.py"; do
  if [[ -f "$cand" ]]; then TARGET="$cand"; break; fi
done
if [[ -z "$TARGET" ]]; then
  echo "❌ 找不到 main_loop.py(试过 models/ , ./ , ../models/)。"
  echo "   请在 code 根目录 或 models/ 目录下运行。"
  exit 1
fi
echo "[target] $TARGET"
BAK="${TARGET}.bak_split"
MARKER="# [v21-SPLIT-ARGS]"

MODE="dryrun"
[[ "${1:-}" == "--apply" ]] && MODE="apply"
[[ "${1:-}" == "--revert" ]] && MODE="revert"

if [[ ! -f "$TARGET" ]]; then
  echo "❌ $TARGET 不存在。"; exit 1
fi

if [[ "$MODE" == "revert" ]]; then
  if [[ -f "$BAK" ]]; then cp "$BAK" "$TARGET"; echo "✅ 已还原 $TARGET"; else echo "❌ 无备份"; exit 1; fi
  exit 0
fi

if grep -qF "$MARKER" "$TARGET"; then
  echo "ℹ️  已含拆分参数, 跳过。重打先 --revert"; exit 0
fi

ANCHOR='    args = parser.parse_args()'
if ! grep -qF "$ANCHOR" "$TARGET"; then
  echo "❌ 找不到锚点 'args = parser.parse_args()'"; exit 1
fi

echo "将在 '$ANCHOR' 之前注入三个拆分参数。"
if [[ "$MODE" == "dryrun" ]]; then
  echo "🔍 DRY-RUN, 未改动。确认后: bash $0 --apply"; exit 0
fi

cp "$TARGET" "$BAK"; echo "💾 备份: $BAK"

python3 - "$TARGET" <<'PYEOF'
import sys
target = sys.argv[1]
anchor = '    args = parser.parse_args()'
inject = '''    # [v21-SPLIT-ARGS] 簇内拆分引擎开关(透传到 run_step2)
    parser.add_argument('--enable_split', action='store_true', default=False,
                        help='[v21] 开启簇内拆分(edit层次聚类二分+consensus门控). '
                             '唯一的真实迭代机制. 默认关=当前行为.')
    parser.add_argument('--split_tau', type=int, default=5,
                        help='[v21] 拆分门控阈值: 两子簇consensus edit>=tau才拆. '
                             'spike实测 tau=5 净+518, 在近似重复间距2之上留安全垫.')
    parser.add_argument('--split_min_size', type=int, default=6,
                        help='[v21] 簇read数<此值不尝试拆.')

'''
with open(target, encoding='utf-8') as f:
    src = f.read()
if anchor not in src:
    print("❌ 锚点丢失"); sys.exit(1)
src = src.replace(anchor, inject + anchor, 1)
with open(target, 'w', encoding='utf-8') as f:
    f.write(src)
print("✅ 注入完成")
PYEOF

python3 -m py_compile "$TARGET" && echo "   ✓ main_loop.py 语法OK" || { echo "❌ 语法错误, 还原"; cp "$BAK" "$TARGET"; exit 1; }
echo "✅ 完成。三轮命令追加 --enable_split 即开启。"