#!/usr/bin/env bash
# patch_enable_split.sh
# =====================
# 把 v21 簇内拆分引擎 (models/cluster_split.py) 接入 step2_runner.py。
#
# 单点注入: 在 `new_labels_np = new_labels.cpu().numpy()` 之后、
#           首个 run_feddna_decode 之前, 调用 split_clusters() 原地拆分 new_labels_np。
# 下游 consensus / 落盘 / 下一轮训练全部自动跟随, 无其它改动。
#
# 开关: getattr(args, 'enable_split', False) —— 默认关, 不开则零行为变化。
#       通过 main_loop 传 --enable_split / --split_tau / --split_min_size 控制。
#
# 安全: 备份 .bak_split / 幂等检查(已打过补丁则跳过) / 默认 dry-run。
#
# 用法:
#   bash patch_enable_split.sh            # dry-run, 只预览
#   bash patch_enable_split.sh --apply    # 实际应用
#   bash patch_enable_split.sh --revert   # 从 .bak_split 还原

set -euo pipefail

TARGET="models/step2_runner.py"
BAK="${TARGET}.bak_split"
MARKER="# [v21-SPLIT-INJECT]"

MODE="dryrun"
[[ "${1:-}" == "--apply" ]] && MODE="apply"
[[ "${1:-}" == "--revert" ]] && MODE="revert"

if [[ ! -f "$TARGET" ]]; then
  echo "❌ 找不到 $TARGET, 请在 code 根目录运行此脚本。"
  exit 1
fi

# ── revert ──
if [[ "$MODE" == "revert" ]]; then
  if [[ -f "$BAK" ]]; then
    cp "$BAK" "$TARGET"
    echo "✅ 已从 $BAK 还原 $TARGET"
  else
    echo "❌ 找不到备份 $BAK"
    exit 1
  fi
  exit 0
fi

# ── 幂等检查 ──
if grep -qF "$MARKER" "$TARGET"; then
  echo "ℹ️  $TARGET 已包含拆分注入 (marker 存在), 跳过。"
  echo "    如需重打, 先 bash patch_enable_split.sh --revert"
  exit 0
fi

# ── 定位锚点 ──
ANCHOR='    new_labels_np = new_labels.cpu().numpy()  # 严格版: 训练/聚类评估/保存用'
if ! grep -qF "$ANCHOR" "$TARGET"; then
  echo "❌ 找不到注入锚点行:"
  echo "   $ANCHOR"
  echo "   step2_runner.py 可能已改动, 请手动确认插入点。"
  exit 1
fi

# ── 要注入的代码块 ──
read -r -d '' INJECT <<'PYEOF' || true

    # [v21-SPLIT-INJECT] 簇内拆分引擎 (唯一的真实迭代机制)
    # 在 new_labels_np 生成后、consensus 解码前, 对 new_labels_np 原地拆分。
    # 默认关 (enable_split=False), 开启后净 +518 success(τ=5, spike 实测)。
    if getattr(args, 'enable_split', False):
        from models.cluster_split import split_clusters
        from models.eval_reconstruction import levenshtein as _split_ed
        _split_tau      = getattr(args, 'split_tau', 5)
        _split_min_size = getattr(args, 'split_min_size', 6)
        _split_ref_len  = getattr(args, 'ref_length', None) or 196
        new_labels_np, _split_stats = split_clusters(
            new_labels_np=new_labels_np,
            flat_real_indices=flat_real_indices,
            data_loader=data_loader,
            levenshtein=_split_ed,
            ref_length=_split_ref_len,
            tau=_split_tau,
            min_split_size=_split_min_size,
            verbose=True,
        )
        # 同步回 eval_labels 轨道(Zone III 已关, 双轨基本一致;
        # 拆分只重组已分配 read, 不产生 -1, 对 read_util 中性)
        try:
            import torch as _torch
            new_labels = _torch.from_numpy(new_labels_np).to(new_labels.device)
            eval_labels = new_labels.clone()
        except Exception as _e:
            print(f"   ⚠️ 拆分后同步 eval_labels 失败(不影响训练轨): {_e}")
PYEOF

echo "════════════════════════════════════════════════════════════════"
echo "  将在以下锚点行之后注入拆分调用:"
echo "    $ANCHOR"
echo "────────────────────────────────────────────────────────────────"
echo "$INJECT"
echo "════════════════════════════════════════════════════════════════"

if [[ "$MODE" == "dryrun" ]]; then
  echo ""
  echo "🔍 DRY-RUN: 未改动文件。确认无误后运行:"
  echo "    bash patch_enable_split.sh --apply"
  exit 0
fi

# ── apply ──
cp "$TARGET" "$BAK"
echo "💾 已备份: $BAK"

python3 - "$TARGET" <<PYEOF
import sys
target = sys.argv[1]
anchor = '    new_labels_np = new_labels.cpu().numpy()  # 严格版: 训练/聚类评估/保存用'
inject = '''
    # [v21-SPLIT-INJECT] 簇内拆分引擎 (唯一的真实迭代机制)
    # 在 new_labels_np 生成后、consensus 解码前, 对 new_labels_np 原地拆分。
    # 默认关 (enable_split=False), 开启后净 +518 success(\u03c4=5, spike 实测)。
    if getattr(args, 'enable_split', False):
        from models.cluster_split import split_clusters
        from models.eval_reconstruction import levenshtein as _split_ed
        _split_tau      = getattr(args, 'split_tau', 5)
        _split_min_size = getattr(args, 'split_min_size', 6)
        _split_ref_len  = getattr(args, 'ref_length', None) or 196
        new_labels_np, _split_stats = split_clusters(
            new_labels_np=new_labels_np,
            flat_real_indices=flat_real_indices,
            data_loader=data_loader,
            levenshtein=_split_ed,
            ref_length=_split_ref_len,
            tau=_split_tau,
            min_split_size=_split_min_size,
            verbose=True,
        )
        try:
            import torch as _torch
            new_labels = _torch.from_numpy(new_labels_np).to(new_labels.device)
            eval_labels = new_labels.clone()
        except Exception as _e:
            print(f"   \u26a0\ufe0f 拆分后同步 eval_labels 失败(不影响训练轨): {_e}")
'''

with open(target, encoding='utf-8') as f:
    src = f.read()

if anchor not in src:
    print("❌ 锚点丢失, 中止"); sys.exit(1)

src = src.replace(anchor, anchor + "\n" + inject, 1)
with open(target, 'w', encoding='utf-8') as f:
    f.write(src)
print("✅ 注入完成")
PYEOF

echo ""
echo "🔧 语法检查:"
python3 -m py_compile "$TARGET" && echo "   ✓ step2_runner.py 语法OK" || {
  echo "   ❌ 语法错误! 自动还原"; cp "$BAK" "$TARGET"; exit 1;
}

echo ""
echo "✅ 补丁应用成功。开启拆分需在 main_loop 跑三轮时传:"
echo "    --enable_split --split_tau 5 --split_min_size 6"
echo "   关闭(回到当前行为): 不传 --enable_split 即可。"
echo "   完全还原文件: bash patch_enable_split.sh --revert"