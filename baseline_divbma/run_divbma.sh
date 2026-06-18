#!/usr/bin/env bash
set -e
READ_TXT="${1:?需要 read.txt 路径}"
WORKDIR="${2:?需要工作目录}"
REF_LEN="${3:-196}"

EVYAT="$WORKDIR/evyat.txt"
OUTDIR="$WORKDIR/divbma_out"
CONSENSUS="$OUTDIR/consensus_divbma.fasta"
DIV="$WORKDIR/Reconstruction/DivBMA"
SRC="$DIV/DividerBMA.cpp"
mkdir -p "$OUTDIR"

echo "━━━ [1/4] read.txt → evyat.txt ━━━"
cd "$WORKDIR"
python3 readtxt_to_evyat.py --read_txt "$READ_TXT" --out "$EVYAT" --ref_len "$REF_LEN"

echo "━━━ [2/4] patch DividerBMA.cpp (加 consensus fasta) ━━━"
python3 - "$SRC" << 'PY'
import sys, shutil
src = sys.argv[1]
code = open(src).read()
if "consensus_divbma_stream" in code:
    print("  已 patch, 跳过"); sys.exit(0)
shutil.copy(src, src + ".bak"); print("  备份:", src+".bak")
code = code.replace(
    'string output_path = argv[2];',
    'string output_path = argv[2];\n'
    '    string consensusFastaPath = (argc >= 4) ? string(argv[3]) : (output_path + "/consensus_divbma.fasta");',
    1)
code = code.replace(
    'output.open(output_path+"/output.txt");',
    'output.open(output_path+"/output.txt");\n'
    '            std::ofstream consensus_divbma_stream;\n'
    '            consensus_divbma_stream.open(consensusFastaPath.c_str());',
    1)
code = code.replace(
    'string recon=matika;',
    'string recon=matika;\n'
    '                consensus_divbma_stream << ">cluster_" << (i-1) << "\\n" << recon << "\\n";',
    1)
open(src,"w").write(code)
print("  patch 完成: 3 处插入")
PY

echo "━━━ [3/4] 编译 DivBMA ━━━"
cd "$DIV"
g++ -std=c++0x -O3 -g3 -Wall -o DividerBMA DividerBMA.cpp
echo "  OK"

echo "━━━ [4/4] 运行 DivBMA ━━━"
"$DIV/DividerBMA" "$EVYAT" "$OUTDIR" "$CONSENSUS" > "$OUTDIR/divbma_progress.log" 2>&1 || true
echo "  consensus: $CONSENSUS"
echo "  cluster 数 (必须==11648):"
grep -c '^>cluster_' "$CONSENSUS" || echo "  ⚠️ 看 divbma_progress.log"
