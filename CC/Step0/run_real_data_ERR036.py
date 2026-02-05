import os
import sys
import subprocess
import array
import gc
from collections import defaultdict, Counter

# ================= 实验配置 =================
DATASET_NAME = "ERR036" 
SOURCE_DIR = "给师妹的clover数据集"
SEQ_LENGTH = 152
CLOVER_PROCESSES = 0 # 自动

# ===========================================

def load_fasta_references(fasta_path):
    print(f"📖 [Ref] 读取参考序列: {os.path.basename(fasta_path)} ...")
    refs = {}
    current_tag = None
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_tag = line[1:]
            else:
                if current_tag:
                    refs[current_tag] = line
    return refs

def stream_clover_results(file_path):
    """流式解析 Clover 结果，兼容各种格式"""
    with open(file_path, 'r') as f:
        buffer = ""
        while True:
            chunk = f.read(1024*1024) # 1MB chunks
            if not chunk: break
            
            # 清洗字符
            cleaned = chunk.replace('[', ' ').replace(']', ' ')\
                           .replace('(', ' ').replace(')', ' ')\
                           .replace(',', ' ').replace("'", " ").replace('"', " ")
            buffer += cleaned
            tokens = buffer.split()
            
            if chunk[-1].isspace() or chunk[-1] in "[](),'\"":
                for t in tokens: yield t
                buffer = ""
            else:
                if len(tokens) > 0:
                    for t in tokens[:-1]: yield t
                    buffer = tokens[-1]
                else: pass
        if buffer.strip(): yield buffer.strip()

def process_real_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "Experiments", f"{DATASET_NAME}_Real")
    
    dir_raw = os.path.join(exp_dir, "01_FormattedInput")
    dir_clover = os.path.join(exp_dir, "02_CloverOut")
    dir_feddna = os.path.join(exp_dir, "03_FedDNA_In")
    dir_temp = os.path.join(exp_dir, "99_Temp") # 临时文件目录
    
    for d in [dir_raw, dir_clover, dir_feddna, dir_temp]:
        os.makedirs(d, exist_ok=True)

    if DATASET_NAME == "ERR036":
        reads_file = "ERR036_tags_reads.txt"
        ref_file = "ERR036_fa.fasta"
    elif DATASET_NAME == "Goldman":
        reads_file = "now_goldman_tags_reads.txt"
        ref_file = "Goldman_fa.fasta"
    
    src_reads_path = os.path.join(SOURCE_DIR, reads_file)
    src_ref_path = os.path.join(SOURCE_DIR, ref_file)
    clover_input_path = os.path.join(dir_raw, "clover_input.txt")
    clover_out_file = os.path.join(dir_clover, "clover_result")

    # === Step 1: 检查或生成输入 ===
    if not os.path.exists(clover_input_path) or os.path.getsize(clover_input_path) < 1024:
        print(f"\n[Step 1] 生成 Clover 输入...")
        with open(src_reads_path, 'r') as fin, open(clover_input_path, 'w') as fout:
            line_idx = 1
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 2: continue
                fout.write(f"{line_idx} {parts[-1]}\n")
                line_idx += 1
    else:
        print(f"\n[Step 1] 输入文件已就绪。")

    # === Step 2: 运行 Clover ===
    # 检查 .txt 后缀
    real_clover_out = clover_out_file + ".txt"
    if not os.path.exists(real_clover_out) or os.path.getsize(real_clover_out) < 1024:
        print(f"\n[Step 2] 运行 Clover...")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(current_dir, "Clover") + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, "-m", "clover.main", "-I", clover_input_path, 
               "-O", clover_out_file, "-L", str(SEQ_LENGTH), "-P", str(CLOVER_PROCESSES), "--no-tag"]
        subprocess.run(cmd, check=True, env=env)
    else:
        print(f"\n[Step 2] Clover 结果已就绪，跳过运行。")

    # === Step 3: 外部排序法生成 FedDNA 格式 (内存安全版) ===
    print(f"\n[Step 3] 解析结果并生成 FedDNA 输入 (External Sort Mode)...")
    
    # 3.1 加载 Cluster Map (Array 存储, ~140MB)
    print("   [3.1] 加载聚类映射到内存数组...")
    # 预估最大行数 (3500万足够了)
    cluster_map = array.array('i') 
    
    # 解析 Clover 输出流
    token_gen = stream_clover_results(real_clover_out)
    try:
        while True:
            idx_str = next(token_gen)
            cid_str = next(token_gen)
            # Clover idx 是 1-based，我们只存 cluster_id
            # 假设输出是顺序的 (1, c1), (2, c2)... 如果不是，这里需要更复杂的逻辑
            # 但 Clover 通常按顺序输出。为了保险，我们用 append，隐含 index=read_idx-1
            cluster_map.append(int(cid_str))
    except StopIteration:
        pass
    
    print(f"      映射加载完毕，共 {len(cluster_map)} 条记录。")

    # 3.2 多数投票确定 Reference (流式)
    print("   [3.2] 扫描原始文件，进行 Reference 投票...")
    cluster_votes = defaultdict(lambda: defaultdict(int)) # {cid: {tag: count}}
    
    with open(src_reads_path, 'r') as f:
        valid_idx = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            tag = parts[0]
            if valid_idx < len(cluster_map):
                cid = cluster_map[valid_idx]
                if cid != -1:
                    cluster_votes[cid][tag] += 1
            valid_idx += 1
            
            if valid_idx % 5000000 == 0:
                print(f"      已投票 {valid_idx} 条...")

    # 结算投票，确定每个 Cluster 的 Ref Seq
    print("      正在结算投票...")
    ref_dict = load_fasta_references(src_ref_path)
    cluster_ref_seqs = {} # {cid: "AGCT..."}
    
    for cid, votes in cluster_votes.items():
        if not votes: continue
        most_common_tag = max(votes, key=votes.get)
        if most_common_tag in ref_dict:
            cluster_ref_seqs[cid] = ref_dict[most_common_tag]
            
    # 释放内存
    del cluster_votes
    del ref_dict
    gc.collect()
    
    # 3.3 生成临时排序文件 (ClusterID \t Sequence)
    print("   [3.3] 生成中间文件用于外部排序...")
    temp_unsorted = os.path.join(dir_temp, "unsorted_reads.txt")
    temp_sorted = os.path.join(dir_temp, "sorted_reads.txt")
    
    with open(src_reads_path, 'r') as fin, open(temp_unsorted, 'w') as fout:
        valid_idx = 0
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            if valid_idx < len(cluster_map):
                cid = cluster_map[valid_idx]
                # 只处理有效的、且找到了 Reference 的 Cluster
                if cid != -1 and cid in cluster_ref_seqs:
                    seq = parts[-1]
                    fout.write(f"{cid}\t{seq}\n")
            valid_idx += 1
            
            if valid_idx % 5000000 == 0:
                print(f"      已预处理 {valid_idx} 条...")

    # 释放 Cluster Map
    del cluster_map
    gc.collect()

    # 3.4 调用 Linux Sort 进行外部排序
    print("   [3.4] 执行外部排序 (Linux Sort)...")
    # -n: 按数值排序, -k1,1: 第一列, -S 50%: 使用50%内存缓冲区
    sort_cmd = f"sort -n -k1,1 -S 50% {temp_unsorted} -o {temp_sorted}"
    subprocess.run(sort_cmd, shell=True, check=True)
    
    # 删除未排序的大文件
    os.remove(temp_unsorted)

    # 3.5 生成最终 Output
    print("   [3.5] 写入最终 FedDNA 格式...")
    out_read = os.path.join(dir_feddna, "read.txt")
    out_ref = os.path.join(dir_feddna, "ref.txt")
    
    current_cid = None
    
    with open(temp_sorted, 'r') as fin, open(out_read, 'w') as fr, open(out_ref, 'w') as ff:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) != 2: continue
            
            cid = int(parts[0])
            seq = parts[1]
            
            # 如果换了 Cluster (新的一块)
            if cid != current_cid:
                # 关闭上一个 (如果是第一个则不关闭)
                if current_cid is not None:
                    fr.write("===============================\n")
                
                # 开始新的 Cluster
                current_cid = cid
                ref_seq = cluster_ref_seqs[cid]
                
                # 写入 Ref (ref.txt 和 read.txt 的头部)
                ff.write(ref_seq + "\n")
                fr.write(ref_seq + "\n")
            
            # 写入当前 Read
            fr.write(seq + "\n")
            
        # 最后一个 Cluster 闭合
        if current_cid is not None:
            fr.write("===============================\n")

    # 清理
    if os.path.exists(temp_sorted):
        os.remove(temp_sorted)

    print("-" * 40)
    print(f"🎉 处理完成！所有文件已生成。")
    print(f"👉 结果位置: {dir_feddna}")
    print("-" * 40)

if __name__ == "__main__":
    process_real_data()