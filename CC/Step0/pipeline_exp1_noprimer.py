import os
import sys
import subprocess
import glob
import shutil
import re
import multiprocessing
from multiprocessing import Pool
from collections import defaultdict

try:
    import edlib
    HAS_EDLIB = True
except ImportError:
    HAS_EDLIB = False
    print("⚠️ 严重警告: 未检测到 edlib！请务必运行 'pip install edlib' 以启用高精度去引物！")

# ================= 实验配置 =================
DATASET_NAME = "exp_1_NoPrimer"
SOURCE_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1"

# === Clover & 切片配置 ===
CHUNK_SIZE = 5000000
CLOVER_PROCESSES = 0
CLOVER_SEQ_LENGTH = 100

# === 引物序列 (Exp 1 专用) ===
FWD = "TCCTGTGCTGCCTGTAATGAGCCAA"
REV = "AGCATAGAACTGAGACCACGGATTG"
# ===========================================


def trim_seq_fuzzy(seq):
    seq = seq.strip()
    if not seq: return ""
    if len(seq) == 150 and seq.startswith(FWD) and seq.endswith(REV):
        return seq[25:-25]
    if HAS_EDLIB:
        res_fwd = edlib.align(FWD, seq[:35], mode="HW", task="locations")
        start = res_fwd['locations'][0][1] + 1 if res_fwd['locations'] else 25
        res_rev = edlib.align(REV, seq[-35:], mode="HW", task="locations")
        end = len(seq) - 35 + res_rev['locations'][-1][0] if res_rev['locations'] else len(seq) - 25
        start = min(max(start, 20), 30)
        end = min(max(end, len(seq) - 30), len(seq) - 20)
        return seq[start:end]
    else:
        return seq[25:-25]


def trim_worker(line_data):
    line_idx, tag, raw_seq = line_data
    clean_seq = trim_seq_fuzzy(raw_seq)
    return line_idx, tag, clean_seq


# ===========================================================================
# [NEW] 碱基级多数投票 consensus
# 完全自监督：只用 reads 本身，不依赖外部 reference
# ===========================================================================
def majority_vote_consensus(seqs, seq_len):
    """
    对一组序列做逐位碱基多数投票。
    Args:
        seqs:    list of str，同一簇的所有 reads
        seq_len: 输出序列长度（取众数最多的那位）
    Returns:
        consensus: str，长度为 seq_len
    """
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    idx_to_base = ['A', 'C', 'G', 'T']

    counts = [[0, 0, 0, 0] for _ in range(seq_len)]
    for seq in seqs:
        for pos in range(min(len(seq), seq_len)):
            b = seq[pos].upper()
            if b in base_to_idx:
                counts[pos][base_to_idx[b]] += 1

    consensus = []
    for pos in range(seq_len):
        total = sum(counts[pos])
        if total == 0:
            consensus.append('A')  # 兜底
        else:
            consensus.append(idx_to_base[counts[pos].index(max(counts[pos]))])
    return ''.join(consensus)


def main_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "Experiments", DATASET_NAME)

    dir_clover = os.path.join(exp_dir, "02_CloverOut")
    dir_feddna = os.path.join(exp_dir, "03_FedDNA_In")
    dir_temp   = os.path.join(exp_dir, "99_Temp")
    dir_chunks = os.path.join(exp_dir, "00_Chunks")

    for d in [dir_clover, dir_feddna, dir_temp, dir_chunks]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    src_reads_path = os.path.join(SOURCE_DIR, "exp1_tags_reads.txt")

    # =========================================================================
    # [Step 1] 多进程高精度去引物 & 生成 Clover 切片
    # =========================================================================
    print(f"\n[Step 1] 🚀 启动多进程高精度去引物并生成 Chunk...")

    tasks = []
    with open(src_reads_path, 'r') as fin:
        for line_idx_0based, line in enumerate(fin):
            parts = line.strip().split()
            if len(parts) >= 2:
                tasks.append((line_idx_0based + 1, parts[0], parts[-1]))

    print(f"   ⏳ 正在清洗 {len(tasks):,} 条 Reads 的引物区，请稍候...")
    with Pool(multiprocessing.cpu_count()) as pool:
        trimmed_results = pool.map(trim_worker, tasks)

    trimmed_full_path = os.path.join(dir_temp, "trimmed_reads_full.txt")
    chunk_idx = 0
    valid_count = 0
    current_out = None

    print(f"   💾 正在写入清洗后的数据切片...")
    with open(trimmed_full_path, 'w') as f_full:
        for line_idx, tag, clean_seq in trimmed_results:
            if len(clean_seq) < 1:
                continue
            f_full.write(f"{line_idx}\t{tag}\t{clean_seq}\n")
            if valid_count % CHUNK_SIZE == 0:
                if current_out: current_out.close()
                chunk_name = os.path.join(dir_chunks, f"chunk_{chunk_idx:03d}.txt")
                current_out = open(chunk_name, 'w')
                chunk_idx += 1
            current_out.write(f"{line_idx} {clean_seq}\n")
            valid_count += 1

    if current_out: current_out.close()
    print(f"   ✅ 去引物完成。有效 reads: {valid_count:,}，生成 {chunk_idx} 个 Chunk。")
    del tasks, trimmed_results

    # =========================================================================
    # [Step 2] 运行 Clover
    # =========================================================================
    print(f"\n[Step 2] 🚀 运行 Clover (L={CLOVER_SEQ_LENGTH})...")
    existing_chunks = sorted(glob.glob(os.path.join(dir_chunks, "chunk_*.txt")))

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(current_dir, "Clover") + os.pathsep + env.get("PYTHONPATH", "")

    final_clover_result = os.path.join(dir_clover, "clover_result_merged.txt")
    with open(final_clover_result, 'w') as f_merged: pass

    for i, chunk_path in enumerate(existing_chunks):
        chunk_name = os.path.basename(chunk_path)
        chunk_out_base = os.path.join(dir_chunks, f"out_{chunk_name}")
        chunk_out_txt  = chunk_out_base + ".txt"
        print(f"   🚀 [{i+1}/{len(existing_chunks)}] 处理: {chunk_name}")
        cmd = [sys.executable, "-m", "clover.main", "-I", chunk_path,
               "-O", chunk_out_base, "-L", str(CLOVER_SEQ_LENGTH),
               "-P", str(CLOVER_PROCESSES), "--no-tag"]
        try:
            subprocess.run(cmd, check=True, env=env)
            with open(chunk_out_txt, 'r') as f_in, open(final_clover_result, 'a') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(chunk_out_txt)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 切片 {chunk_name} 失败! Exit Code: {e.returncode}")
            return

    print(f"   ✅ Clover 聚类完成。")

    # =========================================================================
    # [Step 3] 解析 Clover 输出，收集每个簇的 reads
    # =========================================================================
    print(f"\n[Step 3] 📊 解析 Clover 输出，按簇收集 reads...")

    with open(final_clover_result, 'r') as f:
        content = f.read()
    pairs = re.findall(r"\('?(\d+)'?,\s*'?(\d+)'?\)", content)
    idx_to_cid = {int(idx): int(cid) for idx, cid in pairs}
    print(f"   📍 Clover 成功聚类 reads: {len(idx_to_cid):,} 条")
    del content, pairs

    # 收集每个簇的 clean_seq（用于投票）
    cluster_reads: dict = defaultdict(list)   # {cid: [clean_seq, ...]}
    idx_to_cleanseq = {}

    with open(trimmed_full_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                idx, tag, clean_seq = int(parts[0]), parts[1], parts[2]
                idx_to_cleanseq[idx] = clean_seq
                if idx in idx_to_cid:
                    cid = idx_to_cid[idx]
                    cluster_reads[cid].append(clean_seq)

    print(f"   📍 共 {len(cluster_reads):,} 个簇收集到 reads")

    # =========================================================================
    # [Step 4] 碱基级多数投票生成 ref，写入 FedDNA 输入
    # =========================================================================
    print(f"\n[Step 4] 🗳️ 碱基多数投票生成 ref + 写入 FedDNA 输入...")

    temp_unsorted = os.path.join(dir_temp, "unsorted_reads.txt")
    temp_sorted   = os.path.join(dir_temp, "sorted_reads.txt")

    valid_count = 0
    with open(temp_unsorted, 'w') as fout:
        for idx, cid in idx_to_cid.items():
            if idx in idx_to_cleanseq and cid in cluster_reads:
                fout.write(f"{cid}\t{idx_to_cleanseq[idx]}\n")
                valid_count += 1

    print(f"   🔄 外部排序 {valid_count:,} 条 reads...")
    subprocess.run(f"sort -n -k1,1 -S 50% {temp_unsorted} -o {temp_sorted}",
                   shell=True, check=True)

    out_read   = os.path.join(dir_feddna, "read.txt")
    out_ref    = os.path.join(dir_feddna, "ref.txt")
    out_labels = os.path.join(exp_dir, "clover_labels.txt")

    current_cid   = None
    cluster_count = 0
    read_count    = 0

    # 先算好各簇的 consensus（投票），避免边写边算
    print(f"   🗳️ 正在对 {len(cluster_reads):,} 个簇做碱基多数投票...")
    cluster_consensus = {}
    for cid, seqs in cluster_reads.items():
        seq_len = max(len(s) for s in seqs)
        cluster_consensus[cid] = majority_vote_consensus(seqs, seq_len)
    print(f"   ✅ 投票完成")

    print(f"   💾 写入 read.txt / ref.txt ...")
    with open(temp_sorted, 'r') as fin, \
         open(out_read,   'w') as fr,  \
         open(out_ref,    'w') as ff,  \
         open(out_labels, 'w') as fl:

        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) != 2: continue

            cid = int(parts[0])
            seq = parts[1]

            if cid != current_cid:
                if current_cid is not None:
                    fr.write("===============================\n")
                current_cid = cid
                cluster_count += 1
                # [NEW] 写入碱基多数投票的 consensus，而不是原始 reference
                ff.write(cluster_consensus[cid] + "\n")

            fr.write(seq + "\n")
            fl.write(f"{cluster_count - 1}\n")
            read_count += 1

        if current_cid is not None:
            fr.write("===============================\n")

    print(f"\n🎉 完成！")
    print(f"📊 有效簇数:  {cluster_count:,}")
    print(f"📊 有效 reads: {read_count:,}")
    print(f"📁 输出目录:  {exp_dir}")
    print(f"   ref.txt 来源: 簇内 reads 碱基级多数投票（完全自监督）")


if __name__ == "__main__":
    main_pipeline()