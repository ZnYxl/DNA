"""
fix_id20_step3.py  —  仅重跑 Step 3 (解析 Clover 输出 → 生成 FedDNA_In)

修复:
  1. idx 映射: 用 Clover tuple 的 idx 做字典映射 (非顺序填充)
  2. reads 裁剪: 177bp → ~150bp (去 Illumina adapter, 保留引物, 与 ref 一致)

用法:
  cd /mnt/st_data/liangxinyi/code/CC/Step0
  python fix_id20_step3.py
"""
import os
import re
import subprocess
import shutil
from collections import defaultdict

# ================= 配置 =================
SOURCE_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/id20"
EXP_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/id20_Real"

CLOVER_RESULT = os.path.join(EXP_DIR, "02_CloverOut", "clover_result_merged.txt")
SRC_READS = os.path.join(SOURCE_DIR, "id20_tags_reads.txt")
SRC_REFS = os.path.join(SOURCE_DIR, "id20_refs.fasta")

FEDDNA_DIR = os.path.join(EXP_DIR, "03_FedDNA_In")
TEMP_DIR = os.path.join(EXP_DIR, "99_Temp")

# id20 引物特征
FWD_PRIMER = "AGTGCAACAAGTCAATCCGT"    # 20bp
REV_PRIMER = "AATTGAATGCTTGCTTGCCG"    # 20bp
# 锚点: forward primer 尾部 (更短更灵活)
ANCHOR_FWD = "TCAATCCGT"
# ref 长度
REF_LEN = 150
# ========================================


def extract_model_input(sequence):
    """
    从 177bp read 中提取 ~150bp 给模型:
    方案: 找到 forward primer 锚点, 取锚点前20bp位置开始的 150bp
    这样保留 [fwd_primer + payload + rev_primer], 丢弃 Illumina adapter
    """
    pos = sequence.find(ANCHOR_FWD)
    if pos != -1:
        # 锚点在 forward primer 内, 找到 primer 开头
        # ANCHOR_FWD 是 primer 的尾部, primer 完整是 20bp
        # 所以 primer 起始 = pos - (20 - len(ANCHOR_FWD))
        primer_start = max(0, pos - (len(FWD_PRIMER) - len(ANCHOR_FWD)))
        extracted = sequence[primer_start: primer_start + REF_LEN]
    else:
        # 找不到锚点, 直接取前 150bp
        extracted = sequence[:REF_LEN]

    # 长度对齐
    if len(extracted) < REF_LEN:
        extracted = extracted.ljust(REF_LEN, 'N')
    elif len(extracted) > REF_LEN:
        extracted = extracted[:REF_LEN]

    return extracted


def load_fasta_references(fasta_path):
    print(f"📖 读取参考序列: {os.path.basename(fasta_path)} ...")
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
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                refs[lines[i]] = lines[i + 1]

    print(f"   ✅ {len(refs)} 条参考序列")
    return refs


def main():
    os.makedirs(FEDDNA_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # =========================================================================
    # Step 3.1: 解析 Clover (idx, cid) → 字典映射
    # =========================================================================
    print("\n[3.1] 解析 Clover (idx, cid) 对...")
    with open(CLOVER_RESULT, 'r') as f:
        content = f.read()

    pairs = re.findall(r"\('?(\d+)'?,\s*'?(\d+)'?\)", content)
    print(f"      Clover 非噪声: {len(pairs)} 条")

    idx_to_cid = {}
    for idx_str, cid_str in pairs:
        idx_to_cid[int(idx_str)] = int(cid_str)

    del content, pairs

    # 统计总 reads
    total_reads = 0
    with open(SRC_READS, 'r') as f:
        for _ in f:
            total_reads += 1
    noise_count = total_reads - len(idx_to_cid)
    print(f"      总 reads:   {total_reads}")
    print(f"      有标签:     {len(idx_to_cid)}")
    print(f"      噪声:       {noise_count} ({noise_count / total_reads * 100:.1f}%)")

    # =========================================================================
    # Step 3.2: 投票 Reference
    # =========================================================================
    print("\n[3.2] 投票 Reference...")
    cluster_votes = defaultdict(lambda: defaultdict(int))
    with open(SRC_READS, 'r') as f:
        for line_idx_0based, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            tag = parts[0]
            line_idx_1based = line_idx_0based + 1

            if line_idx_1based in idx_to_cid:
                cid = idx_to_cid[line_idx_1based]
                cluster_votes[cid][tag] += 1

            if (line_idx_0based + 1) % 5000000 == 0:
                print(f"      已投票 {line_idx_0based + 1} 条...", end='\r')

    print(f"\n      结算投票...")
    ref_dict = load_fasta_references(SRC_REFS)
    cluster_ref_seqs = {}
    matched_count = 0
    for cid, votes in cluster_votes.items():
        best_tag = max(votes, key=votes.get)
        if best_tag in ref_dict:
            cluster_ref_seqs[cid] = ref_dict[best_tag]
            matched_count += 1

    print(f"      成功匹配 Reference: {matched_count}")
    del cluster_votes, ref_dict

    # =========================================================================
    # Step 3.3: 生成排序中间文件 (裁剪 reads 至 ~150bp)
    # =========================================================================
    print("\n[3.3] 生成排序中间文件 (reads 裁剪至 150bp)...")
    temp_unsorted = os.path.join(TEMP_DIR, "unsorted_reads.txt")
    temp_sorted = os.path.join(TEMP_DIR, "sorted_reads.txt")

    # 统计锚点命中率
    anchor_hit = 0
    anchor_miss = 0
    written = 0

    with open(SRC_READS, 'r') as fin, open(temp_unsorted, 'w') as fout:
        for line_idx_0based, line in enumerate(fin):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            line_idx_1based = line_idx_0based + 1

            if line_idx_1based in idx_to_cid:
                cid = idx_to_cid[line_idx_1based]
                if cid in cluster_ref_seqs:
                    raw_seq = parts[-1]
                    # 裁剪 177bp → 150bp (去 adapter, 保留引物)
                    model_seq = extract_model_input(raw_seq)

                    if ANCHOR_FWD in raw_seq:
                        anchor_hit += 1
                    else:
                        anchor_miss += 1

                    fout.write(f"{cid}\t{model_seq}\n")
                    written += 1

            if (line_idx_0based + 1) % 5000000 == 0:
                print(f"      已处理 {line_idx_0based + 1} 条...", end='\r')

    del idx_to_cid

    total_anchored = anchor_hit + anchor_miss
    print(f"\n      锚点命中率: {anchor_hit}/{total_anchored} "
          f"({anchor_hit / max(total_anchored, 1) * 100:.1f}%)")
    print(f"      写入 reads: {written}")

    # =========================================================================
    # Step 3.4: 外部排序
    # =========================================================================
    print("\n[3.4] 外部排序...")
    subprocess.run(f"sort -n -k1,1 -S 50% {temp_unsorted} -o {temp_sorted}",
                   shell=True, check=True)
    os.remove(temp_unsorted)

    # =========================================================================
    # Step 3.5: 写入最终文件
    # =========================================================================
    print("\n[3.5] 写入最终 read.txt / ref.txt ...")

    # 备份旧文件
    old_read = os.path.join(FEDDNA_DIR, "read.txt")
    old_ref = os.path.join(FEDDNA_DIR, "ref.txt")
    if os.path.exists(old_read):
        shutil.move(old_read, old_read + ".bak")
    if os.path.exists(old_ref):
        shutil.move(old_ref, old_ref + ".bak")

    out_read = os.path.join(FEDDNA_DIR, "read.txt")
    out_ref = os.path.join(FEDDNA_DIR, "ref.txt")

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

    print(f"\n🎉 id20 Step 3 修复完成！")
    print(f"📊 有效簇:  {cluster_count}")
    print(f"📊 有效 reads: {read_count}")
    print(f"📊 Read 长度: {REF_LEN}bp (与 ref 一致)")

    # 快速验证
    print(f"\n🔍 快速验证...")
    with open(out_read, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            line = line.strip()
            if not line.startswith("="):
                print(f"   read[{i}]: len={len(line)}, head={line[:30]}...")


if __name__ == "__main__":
    main()