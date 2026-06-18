#!/usr/bin/env bash
set -e
READ_TXT="${1:?需要 read.txt 路径}"
WORKDIR="${2:?需要工作目录}"
REF_LEN="${3:-196}"

EVYAT="$WORKDIR/evyat.txt"
OUTDIR="$WORKDIR/bmala_out"
CONSENSUS="$OUTDIR/consensus_bmala.fasta"
BMALA="$WORKDIR/Reconstruction/BMALA"
SRC="$BMALA/BMALookahead.cpp"
mkdir -p "$OUTDIR"

echo "━━━ [1/4] read.txt → evyat.txt ━━━"
cd "$WORKDIR"
python3 readtxt_to_evyat.py --read_txt "$READ_TXT" --out "$EVYAT" --ref_len "$REF_LEN"

echo "━━━ [2/4] patch BMALookahead.cpp (加 consensus fasta) ━━━"
python3 - "$SRC" << 'PY'
import sys, shutil
src = sys.argv[1]
code = open(src).read()
if "consensus_bmala_stream" in code:
    print("  已 patch, 跳过"); sys.exit(0)
shutil.copy(src, src + ".bak"); print("  备份:", src+".bak")
code = code.replace(
    'string output_path = argv[2];',
    'string output_path = argv[2];\n'
    '    string consensusFastaPath = (argc >= 4) ? string(argv[3]) : (output_path + "/consensus_bmala.fasta");',
    1)
code = code.replace(
    'output.open(output_path+"/output.txt");',
    'output.open(output_path+"/output.txt");\n'
    '            std::ofstream consensus_bmala_stream;\n'
    '            consensus_bmala_stream.open(consensusFastaPath.c_str());',
    1)
code = code.replace(
    'string recon=karin_w3;',
    'string recon=karin_w3;\n'
    '                consensus_bmala_stream << ">cluster_" << (i-1) << "\\n" << recon << "\\n";',
    1)
open(src,"w").write(code)
print("  patch 完成: 3 处插入")
PY

echo "━━━ [3/4] 编译 BMALA ━━━"
cd "$BMALA"
g++ -std=c++0x -O3 -g3 -Wall -o BMALookahead BMALookahead.cpp
echo "  OK"

echo "━━━ [4/4] 运行 BMALA ━━━"
"$BMALA/BMALookahead" "$EVYAT" "$OUTDIR" "$CONSENSUS" > "$OUTDIR/bmala_progress.log" 2>&1 || true
echo "  consensus: $CONSENSUS"
echo "  cluster 数 (必须==11648):"
grep -c '^>cluster_' "$CONSENSUS" || echo "  ⚠️ 看 bmala_progress.log"
