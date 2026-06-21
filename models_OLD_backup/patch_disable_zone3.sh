#!/bin/bash
# ============================================================
# patch_disable_zone3.sh
# 关闭 Zone III 标签隔离 (step2_runner.py line 477)
#
# 原因: 诊断确认 SR 退化的真根因是覆盖率(Recall)塌陷 ——
#   Zone III 逐轮把中等簇(均值25 read)的 read 蚕食成 -1,
#   到 R3 这些簇 97% 的 read 变 -1 → 簇消失 → 对应 GT 失去
#   consensus → Recall 从 0.94 跌到 0.90。而重建质量(RQS 0.96)
#   一直稳, EER 还在降 —— 隔离对质量贡献≈0, 对覆盖率伤害巨大。
#
# 本 patch: 注释掉 477 行的 new_labels[zone_ids==3]=-1,
#   让 Zone III read 保留原簇标签, 不被隔离成 -1, 簇不消失。
#   473-475 行的记录代码保留(无副作用)。
#   下游(归巢/保存/decode)均不硬依赖该行, 已 grep 确认安全。
#
# 用法:
#   bash patch_disable_zone3.sh --dry-run   # 预览, 不改文件
#   bash patch_disable_zone3.sh --apply     # 应用 (自动 .bak 备份)
#   bash patch_disable_zone3.sh --revert    # 从 .bak 恢复
# ============================================================
set -e

FILE="/mnt/st_data/liangxinyi/code/models/step2_runner.py"
BAK="${FILE}.bak_zone3"

OLD='    new_labels[zone_ids == 3] = -1'
NEW='    # [DISABLE-ZONE3] 关闭隔离以验证覆盖率假设: 原行 new_labels[zone_ids == 3] = -1
    pass  # Zone III read 保留原簇标签, 不隔离成 -1, 防止簇被蚕食空'

MODE="${1:---dry-run}"

case "$MODE" in
  --dry-run)
    echo "=== DRY RUN (不改文件) ==="
    echo "目标文件: $FILE"
    echo ""
    echo "--- 将匹配并替换的行 (期望唯一) ---"
    if grep -nF "$OLD" "$FILE"; then
      cnt=$(grep -cF "$OLD" "$FILE")
      echo ""
      echo "匹配行数: $cnt (必须=1 才安全)"
      if [ "$cnt" -ne 1 ]; then
        echo "⚠️ 匹配行数≠1, 不要 apply, 先人工确认!"
      fi
    else
      echo "❌ 没找到目标行, 可能行已变化, 请人工检查 line 477 附近"
    fi
    echo ""
    echo "--- 替换为 ---"
    echo "$NEW"
    echo ""
    echo "确认无误后运行: bash $0 --apply"
    ;;

  --apply)
    if [ -f "$BAK" ]; then
      echo "⚠️ 备份已存在 $BAK, 说明可能已 patch 过。先 --revert 或手动检查。"
      exit 1
    fi
    cnt=$(grep -cF "$OLD" "$FILE")
    if [ "$cnt" -ne 1 ]; then
      echo "❌ 目标行匹配 $cnt 次 (需=1), 中止。"
      exit 1
    fi
    cp "$FILE" "$BAK"
    echo "✅ 已备份: $BAK"
    # 用 python 做精确单行替换 (避免 sed 转义问题)
    python3 - "$FILE" << 'PYEOF'
import sys
f = sys.argv[1]
old = '    new_labels[zone_ids == 3] = -1'
new = ('    # [DISABLE-ZONE3] 关闭隔离以验证覆盖率假设: 原行 new_labels[zone_ids == 3] = -1\n'
       '    pass  # Zone III read 保留原簇标签, 不隔离成 -1, 防止簇被蚕食空')
s = open(f).read()
assert s.count(old) == 1, f"匹配 {s.count(old)} 次, 中止"
s = s.replace(old, new)
open(f,'w').write(s)
print("✅ 已替换 line 477")
PYEOF
    echo ""
    echo "--- 验证替换结果 (line 473-480) ---"
    sed -n '473,480p' "$FILE"
    echo ""
    echo "✅ 完成。重跑三轮迭代, 然后用 spike_rqs_calibration.py 量 RQS+Recall。"
    echo "   对比关掉前后: Recall 是否回升, RQS 是否保持。"
    echo "   恢复原状: bash $0 --revert"
    ;;

  --revert)
    if [ ! -f "$BAK" ]; then
      echo "❌ 找不到备份 $BAK, 无法恢复。"
      exit 1
    fi
    cp "$BAK" "$FILE"
    rm "$BAK"
    echo "✅ 已从备份恢复, 并删除 .bak"
    sed -n '473,480p' "$FILE"
    ;;

  *)
    echo "用法: bash $0 [--dry-run|--apply|--revert]"
    exit 1
    ;;
esac