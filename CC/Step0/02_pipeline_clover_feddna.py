import os
import sys
import subprocess
import glob
import shutil
import array
from collections import defaultdict

# ================= 配置区域 =================
# 实验主目录
BASE_EXP_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_Final"
# 输入数据 (来自 Step 1 的输出)
CLEAN_DATA_DIR = os.path.join(BASE_EXP_DIR, "00_CleanData")
SRC_READS = os.path.join(CLEAN_DATA_DIR, "reads_clean.txt")
SRC_REFS = os.path.join(CLEAN_DATA_DIR, "refs_clean.fasta")

# Clover 参数
SEQ_LENGTH = 100  # 已经去过引物了，现在是 100
CHUNK_SIZE = 5000000
CLOVER_PROCESSES = 0
# ===========================================

def load_clean_refs(fasta_path):
    print(f"📖 读取参考序列...")
    refs = {}
    with open(fasta_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if lines[0].startswith(">"):
        curr = None
        for l in lines:
            if l.startswith(">"): curr = l[1:]
            elif curr: refs[curr] = l
    else:
        for i in range(0, len(lines), 2):
            refs[lines[i]] = lines[i+1]
    return refs

def run_pipeline():
    # 目录准备
    dir_chunks = os.path.join(BASE_EXP_DIR, "01_Chunks")
    dir_clover = os.path.join(BASE_EXP_DIR, "02_CloverOut")
    dir_feddna = os.path.join(BASE_EXP_DIR, "03_FedDNA_In")
    dir_temp = os.path.join(BASE_EXP_DIR, "99_Temp")
    
    for d in [dir_chunks, dir_clover, dir_feddna, dir_temp]:
        os.makedirs(d, exist_ok=True)
        
    # === 1. 切片 (Chunking) ===
    print(f"\n[Step 1] 切片 (直接读取干净数据)...")
    if not os.path.exists(SRC_READS):
        print(f"❌ 错误: 找不到 {SRC_READS}，请先运行 01_preprocess_trim.py")
        return

    chunk_idx = 0
    line_count = 0
    current_out = None
    
    # 检查是否已切分
    existing_chunks = sorted(glob.glob(os.path.join(dir_chunks, "chunk_*.txt")))
    if not existing_chunks:
        with open(SRC_READS, 'r') as fin:
            for line in fin:
                if line_count % CHUNK_SIZE == 0:
                    if current_out: current_out.close()
                    chunk_name = os.path.join(dir_chunks, f"chunk_{chunk_idx:03d}.txt")
                    current_out = open(chunk_name, 'w')
                    print(f"   正在生成: chunk_{chunk_idx:03d}.txt ...", end='\r')
                    chunk_idx += 1
                
                parts = line.strip().split()
                if len(parts) >= 2:
                    # 写入格式: 行号 序列 (已经是干净序列了)
                    current_out.write(f"{line_count + 1} {parts[-1]}\n")
                line_count += 1
        if current_out: current_out.close()
        print(f"\n   ✅ 切片完成。")
        existing_chunks = sorted(glob.glob(os.path.join(dir_chunks, "chunk_*.txt")))
    else:
        print(f"   检测到已有切片，跳过。")

    # === 2. 运行 Clover ===
    print(f"\n[Step 2] 运行 Clover...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(current_dir, "Clover") + os.pathsep + env.get("PYTHONPATH", "")
    
    final_clover_result = os.path.join(dir_clover, "clover_result_merged.txt")
    
    # 如果没跑完才跑
    if not os.path.exists(final_clover_result):
        with open(final_clover_result, 'w') as f_merged: pass
        
        for i, chunk_path in enumerate(existing_chunks):
            chunk_name = os.path.basename(chunk_path)
            chunk_out_base = os.path.join(dir_chunks, f"out_{chunk_name}")
            chunk_out_txt = chunk_out_base + ".txt"
            
            print(f"   🚀 [{i+1}/{len(existing_chunks)}] 处理: {chunk_name}")
            cmd = [sys.executable, "-m", "clover.main", "-I", chunk_path, 
                   "-O", chunk_out_base, "-L", str(SEQ_LENGTH), "-P", str(CLOVER_PROCESSES), "--no-tag"]
            subprocess.run(cmd, check=True, env=env)
            
            with open(chunk_out_txt, 'r') as f_in, open(final_clover_result, 'a') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(chunk_out_txt)
    else:
        print(f"   检测到聚类结果已存在，跳过。")

    # === 3. 生成 FedDNA 格式 ===
    print(f"\n[Step 3] 生成 FedDNA 输入...")
    
    # 3.1 加载 Clover 结果
    print("   加载聚类映射...")
    cluster_map = array.array('i')
    with open(final_clover_result, 'r') as f:
        tokens = f.read().replace('[',' ').replace(']',' ').replace(',',' ').split()
        for i in range(0, len(tokens), 2):
            cluster_map.append(int(tokens[i+1]))
            
    # 3.2 投票 Reference
    print("   投票 Reference...")
    votes = defaultdict(lambda: defaultdict(int))
    with open(SRC_READS, 'r') as f:
        for i, line in enumerate(f):
            if i >= len(cluster_map): break
            cid = cluster_map[i]
            if cid != -1:
                parts = line.strip().split()
                if parts: votes[cid][parts[0]] += 1
    
    # 3.3 确定 Ref
    raw_refs = load_clean_refs(SRC_REFS)
    cluster_ref = {}
    for cid, v in votes.items():
        best_tag = max(v, key=v.get)
        if best_tag in raw_refs:
            cluster_ref[cid] = raw_refs[best_tag] # 已经是干净的了

    # 3.4 写入最终文件
    print("   写入最终文件...")
    out_read = os.path.join(dir_feddna, "read.txt")
    out_ref = os.path.join(dir_feddna, "ref.txt")
    
    curr_cid = None
    cluster_cnt = 0
    
    # 直接读 reads_clean.txt 写入，不需要再排序 (Clover输出本身就是按序的映射，但如果需要把同一个簇的放一起，最好还是排个序)
    # 为了保险，我们还是生成临时文件排个序，虽然 reads 顺序和 map 是一致的，但我们要把同一 cluster 的聚合在一起
    
    temp_unsorted = os.path.join(dir_temp, "unsorted.txt")
    temp_sorted = os.path.join(dir_temp, "sorted.txt")
    
    with open(SRC_READS, 'r') as fin, open(temp_unsorted, 'w') as fout:
        for i, line in enumerate(fin):
            if i >= len(cluster_map): break
            cid = cluster_map[i]
            if cid != -1 and cid in cluster_ref:
                # 写入: ClusterID 序列
                parts = line.strip().split()
                fout.write(f"{cid}\t{parts[-1]}\n")
                
    subprocess.run(f"sort -n -k1,1 -S 50% {temp_unsorted} -o {temp_sorted}", shell=True, check=True)
    os.remove(temp_unsorted)
    
    with open(temp_sorted, 'r') as fin, open(out_read, 'w') as fr, open(out_ref, 'w') as ff:
        for line in fin:
            parts = line.strip().split('\t')
            cid = int(parts[0])
            seq = parts[1]
            
            if cid != curr_cid:
                if curr_cid is not None: fr.write("===============================\n")
                curr_cid = cid
                cluster_cnt += 1
                ref_s = cluster_ref[cid]
                ff.write(ref_s + "\n")
                fr.write(ref_s + "\n")
            fr.write(seq + "\n")
            
        if curr_cid is not None: fr.write("===============================\n")
    
    if os.path.exists(temp_sorted): os.remove(temp_sorted)
    print(f"\n🎉 流程结束！")
    print(f"   有效簇数量: {cluster_cnt}")
    print(f"   结果位置: {dir_feddna}")

if __name__ == "__main__":
    run_pipeline()