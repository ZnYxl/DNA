#!/usr/bin/env bash
# =============================================================================
# patch_clean_mode_mainloop.sh
# 软大扫除 · 收尾 (main_loop.py 加 --clean_mode 开关 + 透传)
# -----------------------------------------------------------------------------
# 作用: 两处改动, 让 --clean_mode 从命令行一路透传到 step2_runner
#   ① argparse 区: 在 --enable_split 之前新增 --clean_mode (action store_true)
#   ② step2_args Namespace: 新增 clean_mode=getattr(args,'clean_mode',False)
#
# 透传链: CLI --clean_mode -> args.clean_mode -> step2_args.clean_mode
#         -> step2_runner 内 getattr(args,'clean_mode',False) [前三刀埋的判断]
#
# 防断点: apply 后自动校验 clean_mode 确实进了 step2_args Namespace 区,
#         复刻 [v21-SPLIT-STEP2ARGS] 踩过的"参数传递断点"教训。
#
# clean_mode 默认 False -> 不加 --clean_mode 时行为完全不变, 守 0.9699。
#
# 特性: .bak 备份 + grep 幂等 + --dry-run 预览
# 用法:
#   bash patch_clean_mode_mainloop.sh --dry-run
#   bash patch_clean_mode_mainloop.sh
#   bash patch_clean_mode_mainloop.sh --revert
# =============================================================================
set -euo pipefail

TARGET="/mnt/st_data/liangxinyi/code/models/main_loop.py"
BAK="${TARGET}.bak_cleanmode_mainloop"
MARK="v22-CLEANMODE-MAINLOOP"

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

# ---- 两处 anchor (逐字符匹配, 不靠行号) ----
# ① argparse: 锚定 [v21-SPLIT-ARGS] 注释行, 在它之前插入 --clean_mode 定义
ANCHOR1='    # [v21-SPLIT-ARGS] 簇内拆分引擎开关(透传到 run_step2)'
INSERT1='    # ['"$MARK"'] 大扫除总开关: 显式关闭所有已退役机制, 使拆分成为唯一改 label 的引擎
    parser.add_argument('"'"'--clean_mode'"'"', action='"'"'store_true'"'"', default=False,
                        help='"'"'[v22 大扫除] 关闭死数据复活/归巢/Rebirth 等已退役机制, '"'"'
                             '"'"'使簇内拆分成为唯一改变 label 的迭代引擎. 用于消融: '"'"'
                             '"'"'开启后 SR 应与默认行为一致, 证明其余机制零贡献.'"'"')
'

# ② step2_args: 锚定 [v21-SPLIT-STEP2ARGS] 注释行, 在它之前插入 clean_mode 透传
ANCHOR2='            # [v21-SPLIT-STEP2ARGS] 透传拆分开关到 run_step2'
INSERT2='            # ['"$MARK"'] 透传大扫除总开关到 run_step2
            clean_mode=getattr(args, '"'"'clean_mode'"'"', False),
'

c1=$(grep -F -c "$ANCHOR1" "$TARGET" || true)
c2=$(grep -F -c "$ANCHOR2" "$TARGET" || true)
echo "🔎 anchor 匹配: argparse=$c1  step2_args=$c2 (各期望 1)"
if [ "$c1" -ne 1 ] || [ "$c2" -ne 1 ]; then
  echo "❌ anchor 非唯一匹配。服务器代码可能已变动, 中止以免误改。"
  echo "   请贴回:"
  echo "   grep -n 'v21-SPLIT-ARGS' '$TARGET'"
  echo "   grep -n 'v21-SPLIT-STEP2ARGS' '$TARGET'"
  exit 1
fi

TMP="$(mktemp)"
cp "$TARGET" "$TMP"
python3 - "$TMP" "$ANCHOR1" "$INSERT1" "$ANCHOR2" "$INSERT2" <<'PY'
import sys
path, a1, i1, a2, i2 = sys.argv[1:6]
lines = open(path, encoding='utf-8').read().split('\n')
out, h1, h2 = [], 0, 0
for ln in lines:
    if ln == a1:
        out.extend(i1.split('\n')); h1 += 1
    if ln == a2:
        out.extend(i2.split('\n')); h2 += 1
    out.append(ln)
assert h1 == 1 and h2 == 1, f"anchor 命中 a1={h1} a2={h2}, 期望各1"
open(path, 'w', encoding='utf-8').write('\n'.join(out))
PY

echo ""
echo "===== DIFF 预览 (- 原 / + 改后) ====="
diff -u "$TARGET" "$TMP" || true
echo "====================================="

if [ "$DRY" -eq 1 ]; then
  rm -f "$TMP"
  echo ""
  echo "🧪 dry-run 完成, 原文件未改动。确认两处插入正确后, 去掉 --dry-run 再跑。"
  exit 0
fi

cp "$TARGET" "$BAK"
echo "💾 已备份: $BAK"
cp "$TMP" "$TARGET"
rm -f "$TMP"

# ---- 透传链校验 (防 [v21-SPLIT-STEP2ARGS] 断点重演) ----
echo ""
echo "🔗 透传链校验:"
ok=1
if grep -qF "add_argument('--clean_mode'" "$TARGET"; then
  echo "   ✅ argparse 定义存在"
else
  echo "   ❌ argparse 定义缺失"; ok=0
fi
# clean_mode= 必须出现在 step2_args 的 Namespace 区 (用 enable_split= 作邻近锚)
if grep -qF "clean_mode=getattr(args, 'clean_mode', False)" "$TARGET"; then
  echo "   ✅ step2_args 透传存在"
else
  echo "   ❌ step2_args 透传缺失 — 这正是 [v21-SPLIT-STEP2ARGS] 断点! 请人工检查"; ok=0
fi
# 语法自检
if python3 -c "import ast,sys; ast.parse(open('$TARGET',encoding='utf-8').read())" 2>/dev/null; then
  echo "   ✅ main_loop.py 语法 OK"
else
  echo "   ❌ 语法错误! 立即 --revert"; ok=0
fi

if [ "$ok" -eq 1 ]; then
  echo ""
  echo "✅ 收尾补丁已应用, 透传链完整。"
  grep -n "$MARK" "$TARGET" || true
else
  echo ""
  echo "⚠️ 校验未全通过, 建议 --revert 后人工核对。"
fi

echo ""
echo "下一步: 开 clean_mode 总验证三轮 (命令在原命令基础上加 --clean_mode):"
echo "  python main_loop.py ... --enable_split --split_tau 5 --split_min_size 6 --clean_mode \\"
echo "    2>&1 | tee seq1d_v22_cleanmode_FULL.log"
echo "  预期: SR 仍 0.9539/0.9662/0.9699, 日志里复活/归巢/Rebirth 全部'已禁用/跳过'"
echo "回滚: bash $(basename "$0") --revert"