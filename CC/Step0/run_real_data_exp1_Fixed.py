import os
import sys
import subprocess
import glob
import shutil
import re
from collections import defaultdict

# ================= 实验配置 =================
DATASET_NAME = "exp_1"
SOURCE_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1"

# === 去引物配置 ===
CLOVER_SEQ_LENGTH = 100
ANCHOR_FWD = "GTAATGAGCCAA"

CHUNK_SIZE = 5000000
CLOVER_PROCESSES = 0

# ===========================================


def load_fasta_references(fasta_path):
    print(f"📖 [Ref] 读取参考序列: {os.path.basename(fasta_path)} ...")
    refs = {}
    with open(fasta_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    if lines[0].startswith(">"):
        current_tag = None
        for line in lines:
            if line.startswith(">"):
                current_tag = line[1:]
            elif current_tag:
                refs[current_tag] = line
    else:
        print("   ℹ️ 检测到无 '>' 格式，启用双行读取模式...")
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                tag = lines[i]
                seq = lines[i + 1]
                refs[tag] = seq

    print(f"   ✅ 加载了 {len(refs)} 条参考序列 (Ground Truth)")
    return refs


def extract_payload(sequence):
    """
    针对 exp_1 的去引物逻辑:
    寻找锚点 "GTAATGAGCCAA"，截取其后的 100bp
    """
    pos = sequence.find(ANCHOR_FWD)
    if pos != -1:
        start = pos + len(ANCHOR_FWD)
        payload = sequence[start: start + CLOVER_SEQ_LENGTH]
    else:
        if len(sequence) > 25:
            payload = sequence[25: 25 + CLOVER_SEQ_LENGTH]
        else:
            payload = sequence

    if len(payload) < CLOVER_SEQ_LENGTH:
        payload = payload.ljust(CLOVER_SEQ_LENGTH, 'N')
    elif len(payload) > CLOVER_SEQ_LENGTH:
        payload = payload[:CLOVER_SEQ_LENGTH]

    return payload


def process_exp1_fixed():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "Experiments", f"{DATASET_NAME}_Real_last")

    dir_raw = os.path.join(exp_dir, "01_FormattedInput")
    dir_clover = os.path.join(exp_dir, "02_CloverOut")
    dir_feddna = os.path.join(exp_dir, "03_FedDNA_In")
    dir_temp = os.path.join(exp_dir, "99_Temp")
    dir_chunks = os.path.join(exp_dir, "00_Chunks")

    if os.path.exists(dir_chunks):
        shutil.rmtree(dir_chunks)

    for d in [dir_raw, dir_clover, dir_feddna, dir_temp, dir_chunks]:
        os.makedirs(d, exist_ok=True)

    src_reads_path = os.path.join(SOURCE_DIR, "exp1_tags_reads.txt")
    src_ref_path = os.path.join(SOURCE_DIR, "exp1_refs.fasta")

    # === Step 1: 切片并去引物 (给 Clover 用) ===
    print(f"\n[Step 1] 切片并提取 Payload (去除引物)...")
    if not os.path.exists(src_reads_path):
        print(f"❌ 错误: 找不到源文件 {src_reads_path}")
        return

    chunk_idx = 0
    line_count = 0
    current_out = None

    with open(src_reads_path, 'r') as fin:
        for line in fin:
            if line_count % CHUNK_SIZE == 0:
                if current_out:
                    current_out.close()
                chunk_name = os.path.join(dir_chunks, f"chunk_{chunk_idx:03d}.txt")
                current_out = open(chunk_name, 'w')
                print(f"   正在生成切片: chunk_{chunk_idx:03d}.txt ...", end='\r')
                chunk_idx += 1

            parts = line.strip().split()
            if len(parts) >= 2:
                raw_seq = parts[-1]
                clean_seq = extract_payload(raw_seq)
                global_line_idx = line_count + 1
                current_out.write(f"{global_line_idx} {clean_seq}\n")

            line_count += 1

    if current_out:
        current_out.close()
    print(f"\n   ✅ 切片完成。共 {line_count} 条 reads。")
    existing_chunks = sorted(glob.glob(os.path.join(dir_chunks, "chunk_*.txt")))

    # === Step 2: 逐块运行 Clover ===
    print(f"\n[Step 2] 运行 Clover (L={CLOVER_SEQ_LENGTH})...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(current_dir, "Clover") + os.pathsep + env.get("PYTHONPATH", "")

    final_clover_result = os.path.join(dir_clover, "clover_result_merged.txt")
    if os.path.exists(final_clover_result):
        os.remove(final_clover_result)
    with open(final_clover_result, 'w') as f_merged:
        pass

    for i, chunk_path in enumerate(existing_chunks):
        chunk_name = os.path.basename(chunk_path)
        chunk_out_base = os.path.join(dir_chunks, f"out_{chunk_name}")
        chunk_out_txt = chunk_out_base + ".txt"

        print(f"   🚀 [{i + 1}/{len(existing_chunks)}] 处理切片: {chunk_name}")
        cmd = [sys.executable, "-m", "clover.main", "-I", chunk_path,
               "-O", chunk_out_base, "-L", str(CLOVER_SEQ_LENGTH), "-P", str(CLOVER_PROCESSES), "--no-tag"]
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 切片失败! Exit Code: {e.returncode}")
            return

        with open(chunk_out_txt, 'r') as f_in, open(final_clover_result, 'a') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(chunk_out_txt)

    print(f"   ✅ Clover 聚类完成。")

    # =========================================================================
    # === Step 3: 解析 Clover 输出 (关键修复: 用 idx 做正确映射) ===
    # =========================================================================
    print(f"\n[Step 3] 解析 Clover 输出 (修复版: 按 idx 映射)...")

    # ─── 3.1 解析 Clover tuple 输出 → idx→cid 字典 ───
    print("   [3.1] 解析 Clover (idx, cid) 对...")
    with open(final_clover_result, 'r') as f:
        content = f.read()

    # 匹配 ('idx', 'cid') 或 (idx, cid) 格式
    pairs = re.findall(r"\('?(\d+)'?,\s*'?(\d+)'?\)", content)
    print(f"         Clover 非噪声输出: {len(pairs)} 条")

    # ─── 关键修复 ───
    # Clover 的 idx 是基于 chunk 输入的行号 (1-based)
    # 我们的 chunk 写入时用了 global_line_idx = line_count + 1 (1-based)
    # 所以 idx 就是原始 reads 的 1-based 行号
    idx_to_cid = {}
    for idx_str, cid_str in pairs:
        idx_to_cid[int(idx_str)] = int(cid_str)

    del content, pairs  # 释放内存

    total_reads = line_count
    noise_count = total_reads - len(idx_to_cid)
    print(f"         总 reads: {total_reads}")
    print(f"         有标签:  {len(idx_to_cid)}")
    print(f"         噪声:    {noise_count} ({noise_count / total_reads * 100:.1f}%)")

    # ─── 3.2 投票 Reference ───
    print("   [3.2] 投票 Reference...")
    cluster_votes = defaultdict(lambda: defaultdict(int))
    with open(src_reads_path, 'r') as f:
        for line_idx_0based, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            tag_in_read = parts[0]
            line_idx_1based = line_idx_0based + 1

            if line_idx_1based in idx_to_cid:
                cid = idx_to_cid[line_idx_1based]
                cluster_votes[cid][tag_in_read] += 1

            if (line_idx_0based + 1) % 1000000 == 0:
                print(f"      已投票 {line_idx_0based + 1} 条...", end='\r')

    print(f"\n      结算投票...")
    ref_dict = load_fasta_references(src_ref_path)
    cluster_ref_seqs = {}

    matched_count = 0
    for cid, votes in cluster_votes.items():
        if not votes:
            continue
        best_tag = max(votes, key=votes.get)
        if best_tag in ref_dict:
            cluster_ref_seqs[cid] = ref_dict[best_tag]
            matched_count += 1

    print(f"      成功匹配 Reference: {matched_count}")
    del cluster_votes, ref_dict

    # ─── 3.3 生成排序中间文件 (原始 reads, 含引物) ───
    print("   [3.3] 生成排序中间文件...")
    temp_unsorted = os.path.join(dir_temp, "unsorted_reads.txt")
    temp_sorted = os.path.join(dir_temp, "sorted_reads.txt")

    with open(src_reads_path, 'r') as fin, open(temp_unsorted, 'w') as fout:
        for line_idx_0based, line in enumerate(fin):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            line_idx_1based = line_idx_0based + 1

            if line_idx_1based in idx_to_cid:
                cid = idx_to_cid[line_idx_1based]
                if cid in cluster_ref_seqs:
                    # 写入原始 read (含引物), 不是去引物后的 payload
                    fout.write(f"{cid}\t{parts[-1]}\n")

            if (line_idx_0based + 1) % 1000000 == 0:
                print(f"      已预处理 {line_idx_0based + 1} 条...", end='\r')

    del idx_to_cid

    print("\n   [3.4] 外部排序...")
    subprocess.run(f"sort -n -k1,1 -S 50% {temp_unsorted} -o {temp_sorted}",
                   shell=True, check=True)
    os.remove(temp_unsorted)

    print("   [3.5] 写入最终文件...")
    out_read = os.path.join(dir_feddna, "read.txt")
    out_ref = os.path.join(dir_feddna, "ref.txt")

    current_cid = None
    cluster_count = 0
    read_count = 0
    with open(temp_sorted, 'r') as fin, \
            open(out_read, 'w') as fr, \
            open(out_ref, 'w') as ff:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            cid = int(parts[0])
            seq = parts[1]
            if cid != current_cid:
                if current_cid is not None:
                    fr.write("===============================\n")
                current_cid = cid
                cluster_count += 1
                ref_seq = cluster_ref_seqs[cid]
                ff.write(ref_seq + "\n")
                fr.write(ref_seq + "\n")
            fr.write(seq + "\n")
            read_count += 1
        if current_cid is not None:
            fr.write("===============================\n")

    if os.path.exists(temp_sorted):
        os.remove(temp_sorted)

    print(f"\n🎉 exp_1 处理完毕！")
    print(f"📊 最终有效簇: {cluster_count}")
    print(f"📊 最终有效 reads: {read_count}")


if __name__ == "__main__":
    process_exp1_fixed()