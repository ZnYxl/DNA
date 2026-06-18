#!/usr/bin/env bash
# patch_gtrefs_bareseq.sh
# =======================
# 让 load_gt_refs_fasta 同时兼容两种格式:
#   1. 标准 FASTA(有 '>' 头)         —— 原行为不变
#   2. 裸序列(每行一条序列,无 '>' 头) —— 行号(从1)当 ID
#
# 起因: seq1d_refs.txt / pe_ayb_refs.txt 是裸序列(无 '>' 头),
#       原函数只认 '>' → 返回 0 条 → median 空数组 NaN 崩溃。
#       seq1d 之前能跑是因为用了 reads.fasta(带头), 不是 seq1d_refs.txt。
#
# 对齐说明: eval 用序列做 key 对齐(match_reads_to_gt), refs 的 ID 仅作标识,
#           ID 从几开始不影响 SR/EER 计算, 故裸序列按行号(从1)编号即可。
#
# 安全: 备份 .bak_gtrefs / 幂等 / 默认 dry-run。
# 用法(code根或models/下):
#   bash patch_gtrefs_bareseq.sh           # dry-run
#   bash patch_gtrefs_bareseq.sh --apply
#   bash patch_gtrefs_bareseq.sh --revert

set -euo pipefail

TARGET=""
for cand in "models/eval_reconstruction.py" "eval_reconstruction.py" "../models/eval_reconstruction.py"; do
  if [[ -f "$cand" ]]; then TARGET="$cand"; break; fi
done
if [[ -z "$TARGET" ]]; then
  echo "❌ 找不到 eval_reconstruction.py(试过 models/ , ./ , ../models/)。"; exit 1
fi
echo "[target] $TARGET"
BAK="${TARGET}.bak_gtrefs"
MARKER="# [GTREFS-BARESEQ]"

MODE="dryrun"
[[ "${1:-}" == "--apply" ]] && MODE="apply"
[[ "${1:-}" == "--revert" ]] && MODE="revert"

if [[ "$MODE" == "revert" ]]; then
  if [[ -f "$BAK" ]]; then cp "$BAK" "$TARGET"; echo "✅ 已还原 $TARGET"; else echo "❌ 无备份 $BAK"; exit 1; fi
  exit 0
fi

if grep -qF "$MARKER" "$TARGET"; then
  echo "ℹ️  已含裸序列兼容补丁, 跳过。重打先 --revert"; exit 0
fi

ANCHOR='def load_gt_refs_fasta(path: str) -> dict:'
if ! grep -qF "$ANCHOR" "$TARGET"; then
  echo "❌ 找不到 load_gt_refs_fasta 定义。"; exit 1
fi

echo "将把 load_gt_refs_fasta 替换为'自动检测 FASTA / 裸序列'版本。"
if [[ "$MODE" == "dryrun" ]]; then
  echo "🔍 DRY-RUN, 未改动。确认后: bash $0 --apply"; exit 0
fi

cp "$TARGET" "$BAK"; echo "💾 备份: $BAK"

python3 - "$TARGET" <<'PYEOF'
import sys, re
target = sys.argv[1]
with open(target, encoding='utf-8') as f:
    src = f.read()

# 用正则定位整个 load_gt_refs_fasta 函数体(从 def 到下一个顶层 def 之前)
pattern = re.compile(
    r'def load_gt_refs_fasta\(path: str\) -> dict:.*?(?=\ndef )',
    re.DOTALL
)
m = pattern.search(src)
if not m:
    print("❌ 未能匹配 load_gt_refs_fasta 函数体"); sys.exit(1)

new_func = '''def load_gt_refs_fasta(path: str) -> dict:
    # [GTREFS-BARESEQ] 自动兼容两种格式:
    #   1. 标准 FASTA(含 '>' 头): 按头解析 ID
    #   2. 裸序列(无 '>' 头, 每行一条序列): 行号(从1)当 ID
    # eval 用序列做 key 对齐, ID 仅作标识, 故裸序列按行号编号不影响 SR/EER。
    """从 FASTA 或裸序列文件加载 GT references: {int_id: sequence}。"""
    # 先探测是否含 '>' 头
    has_header = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('>'):
                has_header = True
            break  # 只看第一条非空行

    refs = {}
    if has_header:
        # ---- 标准 FASTA 解析(原逻辑) ----
        cur_id = None
        cur_seq = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if cur_id is not None:
                        refs[cur_id] = ''.join(cur_seq).upper()
                    try:
                        cur_id = int(line[1:].split()[0])
                    except ValueError:
                        cur_id = line[1:].strip()
                    cur_seq = []
                elif line:
                    cur_seq.append(line)
        if cur_id is not None:
            refs[cur_id] = ''.join(cur_seq).upper()
        print(f"   GT references: {len(refs):,}  (FASTA 格式)")
    else:
        # ---- 裸序列解析: 每行一条, 行号(从1)当 ID ----
        with open(path) as f:
            idx = 1
            for line in f:
                seq = line.strip().upper()
                if seq:
                    refs[idx] = seq
                    idx += 1
        print(f"   GT references: {len(refs):,}  (裸序列格式, 行号当 ID)")
    return refs

'''

src = src[:m.start()] + new_func + src[m.end():]
with open(target, 'w', encoding='utf-8') as f:
    f.write(src)
print("✅ 替换完成")
PYEOF

python3 -m py_compile "$TARGET" && echo "   ✓ eval_reconstruction.py 语法OK" || { echo "❌ 语法错误, 还原"; cp "$BAK" "$TARGET"; exit 1; }

echo ""
echo "✅ 完成。现在 load_gt_refs_fasta 同时吃 FASTA 和裸序列。"
echo "   pe_ayb 重建评估可直接用裸序列 refs:"
echo "     --gt_refs .../pe_ayb/pe_ayb_refs.txt"
echo "   还原: bash patch_gtrefs_bareseq.sh --revert"