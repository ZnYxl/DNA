#!/bin/bash
# =============================================================================
# v20.A Patch 1/5: step1_data.py
#   - 新增 seq_to_kmer_bitvec 函数 (5-mer bit vector, 4^5 = 1024 维)
#   - Step1Dataset.__init__ 加 kmer_k 参数 (默认 5)
#   - Step1Dataset.__getitem__ 返回 'kmer_vec' 字段
# =============================================================================
set -e

FILE="${1:-models/step1_data.py}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$FILE" ]]; then
    echo "❌ 文件不存在: $FILE"; exit 1
fi

# ── 健全性检查: anchor 必须只出现一次 ─────────────────────────────
ANCHOR1='from torch.utils.data import Dataset'
ANCHOR2='class Step1Dataset(Dataset):'
ANCHOR3='        cv_threshold: float = 0.3,'
ANCHOR4="        return {
            'encoding': encoding,
            'clover_label': clover_label,
            'gt_label': gt_label,
            'read_idx': real_idx,
            'consensus_target': consensus_target,   # (L, 4) one-hot of pseudo-reference
        }"

for A in "$ANCHOR1" "$ANCHOR2" "$ANCHOR3"; do
    n=$(grep -cF "$A" "$FILE" || true)
    if [[ "$n" != "1" ]]; then
        echo "❌ anchor 出现 $n 次 (应为 1): $A"; exit 1
    fi
done
n=$(grep -cF "kmer_vec" "$FILE" || true)
if [[ "$n" != "0" ]]; then
    echo "⚠️ kmer_vec 已存在, patch 已应用过? 跳过"; exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] 检查通过, 改动如下:"
    echo "  + 新增 seq_to_kmer_bitvec()"
    echo "  + Step1Dataset.__init__: 加 kmer_k 参数 (默认 5)"
    echo "  + Step1Dataset.__getitem__: 返回 'kmer_vec'"
    exit 0
fi

cp "$FILE" "${FILE}.bak_v20A"
echo "📋 备份: ${FILE}.bak_v20A"

# ── 改动 1: 在 imports 后加 seq_to_kmer_bitvec ─────────────────────
python - "$FILE" << 'PYEOF'
import sys, re
path = sys.argv[1]
src = open(path).read()

# anchor: from torch.utils.data import Dataset (单行 import)
anchor = 'from torch.utils.data import Dataset'
inject = '''from torch.utils.data import Dataset


# [v20A] k-mer bit vector for sequence-space Jaccard mask
_KMER_INDEX_CACHE = {}


def _build_kmer_index(k: int):
    """生成 4^k 维的 k-mer 索引表 (A=0, C=1, G=2, T=3)"""
    if k in _KMER_INDEX_CACHE:
        return _KMER_INDEX_CACHE[k]
    base = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    _KMER_INDEX_CACHE[k] = base
    return base


def seq_to_kmer_bitvec(seq: str, k: int = 5):
    """
    返回 (4^k,) 长度的 0/1 torch.float32 tensor.
    位 i 为 1 表示该 read 含有 base-4 编码为 i 的 k-mer.
    Jaccard(s_i, s_j) = (v_i & v_j).sum() / (v_i | v_j).sum() ≈ minhash-style.
    
    [v20A] N 等非 ACGT 字符直接跳过该 k-mer.
    """
    import torch
    base = _build_kmer_index(k)
    dim = 4 ** k
    vec = torch.zeros(dim, dtype=torch.float32)
    L = len(seq)
    for i in range(L - k + 1):
        kmer = seq[i:i + k]
        idx = 0
        valid = True
        for c in kmer:
            if c not in base:
                valid = False
                break
            idx = idx * 4 + base[c]
        if valid:
            vec[idx] = 1.0
    return vec
'''
src = src.replace(anchor, inject, 1)

# ── 改动 2: Step1Dataset.__init__ 加 kmer_k 参数 ──────────────
old_init = '''        cv_threshold: float = 0.3,
                 max_reads_per_cluster: int = 0):'''
new_init = '''        cv_threshold: float = 0.3,
                 max_reads_per_cluster: int = 0,
                 kmer_k: int = 5):  # [v20A] sequence-space Jaccard k-mer size'''
assert src.count(old_init) == 1, f"old_init anchor not unique: {src.count(old_init)}"
src = src.replace(old_init, new_init)

# 在 self.consensus_dict = ... 后面紧接着加 self.kmer_k
old_kk = '        self.consensus_dict = consensus_dict or {}'
new_kk = '''        self.consensus_dict = consensus_dict or {}
        self.kmer_k = kmer_k  # [v20A]'''
assert src.count(old_kk) == 1
src = src.replace(old_kk, new_kk)

# ── 改动 3: __getitem__ return 字典加 'kmer_vec' ──────────────
old_ret = """        return {
            'encoding': encoding,
            'clover_label': clover_label,
            'gt_label': gt_label,
            'read_idx': real_idx,
            'consensus_target': consensus_target,   # (L, 4) one-hot of pseudo-reference
        }"""
new_ret = """        # [v20A] sequence-space k-mer bit vector for Jaccard mask
        kmer_vec = seq_to_kmer_bitvec(seq, k=self.kmer_k)

        return {
            'encoding': encoding,
            'clover_label': clover_label,
            'gt_label': gt_label,
            'read_idx': real_idx,
            'consensus_target': consensus_target,   # (L, 4) one-hot of pseudo-reference
            'kmer_vec': kmer_vec,                   # [v20A] (4^k,) float32 bitvec
        }"""
assert src.count(old_ret) == 1, f"old_ret anchor not unique: {src.count(old_ret)}"
src = src.replace(old_ret, new_ret)

open(path, 'w').write(src)
print("✅ step1_data.py patched")
PYEOF

# ── 验证 ─────────────────────────────────────────────────────────
echo
echo "── grep 验证 ──"
grep -n "seq_to_kmer_bitvec\|kmer_k\|kmer_vec" "$FILE" | head -10
echo
echo "✅ Patch 1/5 done"