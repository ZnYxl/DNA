#!/usr/bin/env bash
set -e
READ_TXT="${1:?需要 read.txt 路径}"
WORKDIR="${2:?需要工作目录}"
REF_LEN="${3:-196}"

ITER_DIR="$WORKDIR/Reconstruction/Iterative"
EVYAT="$WORKDIR/evyat.txt"
OUTDIR="$WORKDIR/iter_out"
CONSENSUS="$OUTDIR/consensus_iter.fasta"
mkdir -p "$OUTDIR"

echo "━━━ [1/4] 编译 Iterative (C++) ━━━"
cd "$ITER_DIR"
rm -f *.o DNA
for src in LCS2 EditDistance Clone Cluster2 LongestPath CommonSubstring2 DividerBMA DNA; do
    g++ -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 -o ${src}.o ${src}.cpp
done
g++ -o DNA *.o
echo "  OK: $ITER_DIR/DNA"

echo "━━━ [2/4] read.txt → evyat.txt ━━━"
cd "$WORKDIR"
python3 readtxt_to_evyat.py --read_txt "$READ_TXT" --out "$EVYAT" --ref_len "$REF_LEN"

echo "━━━ [3/4] 运行 Iterative Reconstruction ━━━"
"$ITER_DIR/DNA" "$EVYAT" "$OUTDIR" "$CONSENSUS" > "$OUTDIR/iter_progress.log" 2>&1 || true
echo "  consensus: $CONSENSUS"
echo "  cluster 数:"
grep -c '^>cluster_' "$CONSENSUS" || echo "  ⚠️ 无输出, 看 iter_progress.log"

echo "━━━ [4/4] 完成, 用 eval_reconstruction.py 评估这个 consensus ━━━"
echo "  consensus fasta: $CONSENSUS"
