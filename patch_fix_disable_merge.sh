#!/usr/bin/env bash
# patch_fix_disable_merge.sh
# ==========================
# 修复 R2/R3 回退的真正根因:
#
# 诊断链(反事实spike + R2探针铁证):
#   - R2 探针: [MNN合并] merges=1198, wrong=301, precision=0.7487, contaminated=15355
#   - 反事实spike: 还原"质心重分配"(实为错误合并) → SR 0.9149→0.9579 (+4.30pt)
#   - 根因: 拆分把簇数推到15663 > target_clusters(11736) → 触发"课程合并" →
#           merge_close_centroids 以25%错误率把R1拆好的纯簇又错误合并 → SR暴跌
#   - disable_merge 开关是坏的: 只打印"已开启"却没真正跳过第~397行的合并调用。
#     R1侥幸没合并是因为簇数(11514)未超target, 不是开关生效。
#
# 修复: disable_merge=True 时, 真正跳过 merge_close_centroids 调用,
#       labels/centroids/cluster_sizes 原样保留(拆分结果不被合并污染)。
#
# 安全: 备份 .bak_mergefix / 幂等 / 默认 dry-run。
# 用法(code根或models/下):
#   bash patch_fix_disable_merge.sh           # dry-run
#   bash patch_fix_disable_merge.sh --apply
#   bash patch_fix_disable_merge.sh --revert

set -euo pipefail

TARGET=""
for cand in "models/step2_runner.py" "step2_runner.py" "../models/step2_runner.py"; do
  if [[ -f "$cand" ]]; then TARGET="$cand"; break; fi
done
if [[ -z "$TARGET" ]]; then
  echo "❌ 找不到 step2_runner.py(试过 models/ , ./ , ../models/)。"; exit 1
fi
echo "[target] $TARGET"
BAK="${TARGET}.bak_mergefix"
MARKER="# [v21-MERGEFIX]"

MODE="dryrun"
[[ "${1:-}" == "--apply" ]] && MODE="apply"
[[ "${1:-}" == "--revert" ]] && MODE="revert"

if [[ "$MODE" == "revert" ]]; then
  if [[ -f "$BAK" ]]; then cp "$BAK" "$TARGET"; echo "✅ 已还原 $TARGET"; else echo "❌ 无备份 $BAK"; exit 1; fi
  exit 0
fi

if grep -qF "$MARKER" "$TARGET"; then
  echo "ℹ️  已含 merge 修复, 跳过。重打先 --revert"; exit 0
fi

# 锚点: merge_close_centroids 调用的起始行
ANCHOR="    centroids, labels_tensor, merge_stats, cluster_sizes = merge_close_centroids("
if ! grep -qF "$ANCHOR" "$TARGET"; then
  echo "❌ 找不到 merge_close_centroids 调用锚点。step2_runner 可能已改, 请手动确认。"; exit 1
fi

echo "将把 merge_close_centroids(...) 调用包进 'if not disable_merge:', "
echo "disable_merge=True 时跳过合并、保留拆分后的 labels/centroids。"
if [[ "$MODE" == "dryrun" ]]; then
  echo "🔍 DRY-RUN, 未改动。确认后: bash $0 --apply"; exit 0
fi

cp "$TARGET" "$BAK"; echo "💾 备份: $BAK"

python3 - "$TARGET" <<'PYEOF'
import sys, re
target = sys.argv[1]
with open(target, encoding='utf-8') as f:
    lines = f.readlines()

anchor = "    centroids, labels_tensor, merge_stats, cluster_sizes = merge_close_centroids("
# 找到锚点行
start = None
for i, ln in enumerate(lines):
    if ln.rstrip('\n') == anchor:
        start = i; break
if start is None:
    print("❌ 锚点行未精确匹配(缩进?)"); sys.exit(1)

# 找到调用结束的 ')' (匹配到首个单独 '    )' 行)
end = None
for j in range(start+1, min(start+40, len(lines))):
    if lines[j].rstrip('\n') == "    )":
        end = j; break
if end is None:
    print("❌ 未找到 merge 调用结束的 ')' 行"); sys.exit(1)

# 构造替换: 把 start..end 这段(含)缩进一级, 外面包 if not disable_merge,
# else 分支补 merge_stats 占位(下游可能引用)。
call_block = lines[start:end+1]
indented = ["    " + ln if ln.strip() else ln for ln in call_block]

new_block = []
new_block.append("    # [v21-MERGEFIX] disable_merge 时真正跳过合并(堵死拆分触发课程合并污染)\n")
new_block.append("    if getattr(args, 'disable_merge', False):\n")
new_block.append("        print(f\"   🚫 [v21-MERGEFIX] disable_merge 生效: 真正跳过 MNN 合并, \"\n")
new_block.append("              f\"保留拆分后 {len(centroids)} 簇不被合并污染\")\n")
new_block.append("        from collections import Counter as _Cmf\n")
new_block.append("        _lc = _Cmf(labels_tensor.tolist())\n")
new_block.append("        cluster_sizes = {cid: _lc.get(cid, 0) for cid in centroids}\n")
new_block.append("        merge_stats = {'clusters_before': len(centroids),\n")
new_block.append("                       'clusters_after': len(centroids),\n")
new_block.append("                       'n_merges': 0, 'time_seconds': 0.0,\n")
new_block.append("                       'threshold': merge_threshold, 'max_cluster_size': 2000}\n")
new_block.append("    else:\n")
new_block.extend(indented)

lines = lines[:start] + new_block + lines[end+1:]
with open(target, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print(f"✅ 注入完成 (原调用行 {start+1}-{end+1})")
PYEOF

python3 -m py_compile "$TARGET" && echo "   ✓ step2_runner.py 语法OK" || { echo "❌ 语法错误, 还原"; cp "$BAK" "$TARGET"; exit 1; }

echo ""
echo "✅ 修复完成。现在 disable_merge=True 时真正跳过合并。"
echo "   重跑三轮(命令不变, 已带 --disable_merge --enable_split):"
echo "   预期: R2/R3 簇数继续由拆分增长(不被合并压回), SR 守住 R1 的 0.95+。"
echo "   还原: bash patch_fix_disable_merge.sh --revert"