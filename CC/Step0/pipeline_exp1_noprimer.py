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
# ⚠️ 注意：去引物后不再是强制的 L=100，Clover 可以处理轻微浮动的序列，设为 100 作为基准
CLOVER_SEQ_LENGTH = 100 

# === 引物序列 (Exp 1 专用) ===
FWD = "TCCTGTGCTGCCTGTAATGAGCCAA"
REV = "AGCATAGAACTGAGACCACGGATTG"
# ===========================================

def trim_seq_fuzzy(seq):
    """使用 edlib 进行高精度动态规划去引物"""
    seq = seq.strip()
    if not seq: return ""
    
    # 对 Reference 等完美序列的极速通道
    if len(seq) == 150 and seq.startswith(FWD) and seq.endswith(REV):
        return seq[25:-25]
        
    if HAS_EDLIB:
        res_fwd = edlib.align(FWD, seq[:35], mode="HW", task="locations")
        start = res_fwd['locations'][0][1] + 1 if res_fwd['locations'] else 25
        
        res_rev = edlib.align(REV, seq[-35:], mode="HW", task="locations")
        end = len(seq) - 35 + res_rev['locations'][-1][0] if res_rev['locations'] else len(seq) - 25
        
        start = min(max(start, 20), 30) 
        # 🌟 修复：REV 限制在 L-30 ~ L-20 之间 (将外层的 max 改成了 min，内层的 min 改成了 max)
        end = min(max(end, len(seq) - 30), len(seq) - 20)
        return seq[start:end]
    else:
        return seq[25:-25]

def trim_worker(line_data):
    """多进程 Worker: 接收 (line_idx, tag, raw_seq)，返回清洗后的元组"""
    line_idx, tag, raw_seq = line_data
    clean_seq = trim_seq_fuzzy(raw_seq)
    return line_idx, tag, clean_seq

def load_and_trim_fasta_references(fasta_path):
    print(f"📖 [Ref] 读取并精准去除参考序列的引物: {os.path.basename(fasta_path)} ...")
    refs = {}
    with open(fasta_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    # 解析 FASTA
    raw_refs = {}
    if lines[0].startswith(">"):
        current_tag = None
        for line in lines:
            if line.startswith(">"):
                current_tag = line[1:]
            elif current_tag:
                raw_refs[current_tag] = line
    else:
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                raw_refs[lines[i]] = lines[i + 1]

    # 去引物
    for tag, seq in raw_refs.items():
        refs[tag] = trim_seq_fuzzy(seq)
        
    print(f"   ✅ 加载并切除了 {len(refs)} 条参考序列 (纯净 Payload ~100bp)")
    return refs

def main_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "Experiments", DATASET_NAME)

    dir_clover = os.path.join(exp_dir, "02_CloverOut")
    dir_feddna = os.path.join(exp_dir, "03_FedDNA_In")
    dir_temp = os.path.join(exp_dir, "99_Temp")
    dir_chunks = os.path.join(exp_dir, "00_Chunks")

    for d in [dir_clover, dir_feddna, dir_temp, dir_chunks]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    src_reads_path = os.path.join(SOURCE_DIR, "exp1_tags_reads.txt")
    src_ref_path = os.path.join(SOURCE_DIR, "exp1_refs.fasta")

    # =========================================================================
    # [Step 1] 多进程高精度去引物 & 生成 Clover 切片
    # =========================================================================
    print(f"\n[Step 1] 🚀 启动多进程高精度去引物并生成 Chunk...")
    
    # 准备任务数据
    tasks = []
    with open(src_reads_path, 'r') as fin:
        for line_idx_0based, line in enumerate(fin):
            parts = line.strip().split()
            if len(parts) >= 2:
                tasks.append((line_idx_0based + 1, parts[0], parts[-1]))

    # 并行处理去引物
    print(f"   ⏳ 正在清洗 {len(tasks):,} 条 Reads 的引物区，请稍候...")
    with Pool(multiprocessing.cpu_count()) as pool:
        trimmed_results = pool.map(trim_worker, tasks)

    # 写入全量中间文件(带Tag) 和 Clover Chunk
    trimmed_full_path = os.path.join(dir_temp, "trimmed_reads_full.txt")
    
    chunk_idx = 0
    valid_count = 0  # 🌟 新增：记录有效 reads 数量
    current_out = None
    
    print(f"   💾 正在写入清洗后的数据切片...")
    with open(trimmed_full_path, 'w') as f_full:
        for line_idx, tag, clean_seq in trimmed_results:
            
            # 🌟 核心修复：过滤掉去引物后变成空字符串或严重残缺的垃圾序列
            if len(clean_seq) < 1: 
                continue
                
            # 记录全量映射供后面使用
            f_full.write(f"{line_idx}\t{tag}\t{clean_seq}\n")
            
            # 写入 Clover Chunk
            if valid_count % CHUNK_SIZE == 0:
                if current_out: current_out.close()
                chunk_name = os.path.join(dir_chunks, f"chunk_{chunk_idx:03d}.txt")
                current_out = open(chunk_name, 'w')
                chunk_idx += 1
                
            current_out.write(f"{line_idx} {clean_seq}\n")
            valid_count += 1  # 🌟 只有有效的才加 1

    if current_out: current_out.close()
    print(f"   ✅ 去引物与切片完成。生成了 {chunk_idx} 个 Chunk。")
    del tasks, trimmed_results # 释放内存

    # =========================================================================
    # [Step 2] 运行 Clover (在纯净的 Payload 上)
    # =========================================================================
    print(f"\n[Step 2] 🚀 运行 Clover (纯净版, L={CLOVER_SEQ_LENGTH})...")
    existing_chunks = sorted(glob.glob(os.path.join(dir_chunks, "chunk_*.txt")))
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(current_dir, "Clover") + os.pathsep + env.get("PYTHONPATH", "")

    final_clover_result = os.path.join(dir_clover, "clover_result_merged.txt")
    with open(final_clover_result, 'w') as f_merged: pass

    for i, chunk_path in enumerate(existing_chunks):
        chunk_name = os.path.basename(chunk_path)
        chunk_out_base = os.path.join(dir_chunks, f"out_{chunk_name}")
        chunk_out_txt = chunk_out_base + ".txt"

        print(f"   🚀 [{i + 1}/{len(existing_chunks)}] 处理切片: {chunk_name}")
        cmd = [sys.executable, "-m", "clover.main", "-I", chunk_path,
               "-O", chunk_out_base, "-L", str(CLOVER_SEQ_LENGTH), "-P", str(CLOVER_PROCESSES), "--no-tag"]
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
    # [Step 3] 解析 Clover 输出与多数投票 (Mapped by idx)
    # =========================================================================
    print(f"\n[Step 3] 📊 解析 Clover 输出并进行 Reference 投票...")

    with open(final_clover_result, 'r') as f:
        content = f.read()
    pairs = re.findall(r"\('?(\d+)'?,\s*'?(\d+)'?\)", content)
    
    idx_to_cid = {int(idx): int(cid) for idx, cid in pairs}
    print(f"   📍 Clover 成功聚类 reads: {len(idx_to_cid):,} 条")
    del content, pairs

    # 投票
    cluster_votes = defaultdict(lambda: defaultdict(int))
    idx_to_cleanseq = {} # 用于 Step 4 生成文件
    
    with open(trimmed_full_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                idx, tag, clean_seq = int(parts[0]), parts[1], parts[2]
                idx_to_cleanseq[idx] = clean_seq
                if idx in idx_to_cid:
                    cid = idx_to_cid[idx]
                    cluster_votes[cid][tag] += 1

    ref_dict = load_and_trim_fasta_references(src_ref_path)
    cluster_ref_seqs = {}
    
    for cid, votes in cluster_votes.items():
        if not votes: continue
        best_tag = max(votes, key=votes.get)
        if best_tag in ref_dict:
            cluster_ref_seqs[cid] = ref_dict[best_tag]

    print(f"   📍 成功映射 Reference 的簇数量: {len(cluster_ref_seqs):,}")

    # =========================================================================
    # [Step 4] 格式化为 FedDNA 输入 (核心修复：写入去引物后的 Payload)
    # =========================================================================
    print(f"\n[Step 4] 📝 生成 FedDNA 输入格式 (按 cid 排序)...")
    
    temp_unsorted = os.path.join(dir_temp, "unsorted_reads.txt")
    temp_sorted = os.path.join(dir_temp, "sorted_reads.txt")

    # 将聚类成功的 clean_seq 写入未排序文件
    valid_count = 0
    with open(temp_unsorted, 'w') as fout:
        for idx, cid in idx_to_cid.items():
            if cid in cluster_ref_seqs and idx in idx_to_cleanseq:
                clean_seq = idx_to_cleanseq[idx]
                fout.write(f"{cid}\t{clean_seq}\n")  # ⚠️ 致命修复：写入 clean_seq 而不是含引物的原序列
                valid_count += 1
                
    print(f"   🔄 开始外部排序 {valid_count:,} 条 reads...")
    subprocess.run(f"sort -n -k1,1 -S 50% {temp_unsorted} -o {temp_sorted}", shell=True, check=True)

    out_read = os.path.join(dir_feddna, "read.txt")
    out_ref = os.path.join(dir_feddna, "ref.txt")
    
    # 顺便生成一个 clover_labels.txt 给后面的主线使用
    out_labels = os.path.join(exp_dir, "clover_labels.txt") 

    current_cid = None
    cluster_count = 0
    read_count = 0
    
    print(f"   💾 正在写入最终的 read.txt (包含分隔符) 和 ref.txt ...")
    with open(temp_sorted, 'r') as fin, \
         open(out_read, 'w') as fr, \
         open(out_ref, 'w') as ff, \
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
                ref_seq = cluster_ref_seqs[cid]
                
                # Ref.txt 只在簇切换时写一次
                ff.write(ref_seq + "\n")
                
            # 写入当前 read (已去引物)
            fr.write(seq + "\n")
            # 记录 label 给你的 DataLoader 初始化使用
            fl.write(f"{cluster_count - 1}\n") 
            read_count += 1
            
        if current_cid is not None:
            fr.write("===============================\n")

    print(f"\n🎉 极其完美！端到端处理完毕！")
    print(f"📊 最终输出文件夹: {exp_dir}")
    print(f"📊 最终纯净有效簇: {cluster_count:,}")
    print(f"📊 最终纯净 reads: {read_count:,}")

if __name__ == "__main__":
    main_pipeline()