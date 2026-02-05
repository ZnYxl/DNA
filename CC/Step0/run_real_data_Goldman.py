import os
import sys
import subprocess
import array
import gc
from collections import defaultdict, Counter

# ================= 实验配置 =================
DATASET_NAME = "Goldman"  # 数据集名称
SOURCE_DIR = "给师妹的clover数据集"
SEQ_LENGTH = 117          # Goldman 数据集的序列长度
CLOVER_PROCESSES = 0      # 0 表示自动使用所有核心

# ===========================================

def load_fasta_references(fasta_path):
    print(f"📖 [Ref] 读取参考序列: {os.path.basename(fasta_path)} ...")
    refs = {}
    current_tag = None
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 兼容标准的 FASTA (>ID) 和你可能遇到的纯文本 ID
            if line.startswith(">"):
                current_tag = line[1:]
            # 如果这一行全是数字/字母且很短（不像DNA序列），可能是不带>的ID
            elif len(line) < 50 and set(line).issubset(set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")):
                 # 这只是一个启发式判断，防止误判短序列。标准FASTA应该有>
                 # 如果你的文件确实没有>，这里会尝试捕获ID
                 # 但根据之前的记录，Goldman_fa.fasta 应该是标准的，所以这里主要依靠 >
                 pass 
            else:
                # 认为是序列
                if current_tag:
                    refs[current_tag] = line
    print(f"   ✅ 加载了 {len(refs)} 条参考序列")
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

def process_goldman_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用独立的文件夹 Goldman_Real
    exp_dir = os.path.join(current_dir, "Experiments", f"{DATASET_NAME}_Real")
    
    dir_raw = os.path.join(exp_dir, "01_FormattedInput")
    dir_clover = os.path.join(exp_dir, "02_CloverOut")
    dir_feddna = os.path.join(exp_dir, "03_FedDNA_In")
    dir_temp = os.path.join(exp_dir, "99_Temp") # 临时文件目录
    
    for d in [dir_raw, dir_clover, dir_feddna, dir_temp]:
        os.makedirs(d, exist_ok=True)

    # Goldman 特定的文件名
    reads_file = "now_goldman_tags_reads.txt"
    ref_file = "Goldman_fa.fasta"
    
    src_reads_path = os.path.join(SOURCE_DIR, reads_file)
    src_ref_path = os.path.join(SOURCE_DIR, ref_file)
    clover_input_path = os.path.join(dir_raw, "clover_input.txt")
    clover_out_file = os.path.join(dir_clover, "clover_result")

    # === Step 1: 检查或生成输入 ===
    if not os.path.exists(clover_input_path) or os.path.getsize(clover_input_path) < 1024:
        print(f"\n[Step 1] 生成 Clover 输入 (Goldman)...")
        if not os.path.exists(src_reads_path):
            print(f"❌ 错误: 找不到源文件 {src_reads_path}")
            return

        with open(src_reads_path, 'r') as fin, open(clover_input_path, 'w') as fout:
            line_idx = 1
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 2: continue
                # Goldman 格式: Tag Sequence
                # 注意：parts[-1] 自动取最后一个非空字段作为序列，兼容中间可能的空格
                fout.write(f"{line_idx} {parts[-1]}\n")
                line_idx += 1
                
                if line_idx % 1000000 == 0:
                    print(f"   已格式化 {line_idx} 条...", end='\r')
        print(f"\n   ✅ 格式化完成。")
    else:
        print(f"\n[Step 1] 输入文件已就绪。")

    # === Step 2: 运行 Clover ===
    real_clover_out = clover_out_file + ".txt"
    if not os.path.exists(real_clover_out) or os.path.getsize(real_clover_out) < 1024:
        print(f"\n[Step 2] 运行 Clover (L={SEQ_LENGTH})...")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(current_dir, "Clover") + os.pathsep + env.get("PYTHONPATH", "")
        
        # 注意: -L 参数已改为 117
        cmd = [sys.executable, "-m", "clover.main", "-I", clover_input_path, 
               "-O", clover_out_file, "-L", str(SEQ_LENGTH), "-P", str(CLOVER_PROCESSES), "--no-tag"]
        
        try:
            print(f"   执行命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"❌ Clover 运行失败: {e}")
            return
    else:
        print(f"\n[Step 2] Clover 结果已就绪，跳过运行。")

    # === Step 3: 外部排序法生成 FedDNA 格式 ===
    print(f"\n[Step 3] 解析结果并生成 FedDNA 输入 (External Sort Mode)...")
    
    # 3.1 加载 Cluster Map (内存优化)
    print("   [3.1] 加载聚类映射...")
    cluster_map = array.array('i') 
    
    token_gen = stream_clover_results(real_clover_out)
    try:
        while True:
            idx_str = next(token_gen) # 消耗掉索引
            cid_str = next(token_gen) # 获取 Cluster ID
            cluster_map.append(int(cid_str))
    except StopIteration:
        pass
    
    print(f"      映射加载完毕，共 {len(cluster_map)} 条记录。")

    # 3.2 多数投票确定 Reference
    print("   [3.2] 扫描原始文件，进行 Reference 投票...")
    cluster_votes = defaultdict(lambda: defaultdict(int)) 
    
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
            
            if valid_idx % 1000000 == 0:
                print(f"      已投票 {valid_idx} 条...", end='\r')

    print("\n      正在结算投票...")
    ref_dict = load_fasta_references(src_ref_path)
    cluster_ref_seqs = {} 
    
    # 修正逻辑：如果 ref_dict 为空（比如FASTA解析失败），这里会报警
    if not ref_dict:
        print("⚠️ 警告：参考序列字典为空！请检查 Goldman_fa.fasta 格式。")
        print("   如果该文件没有 > 符号，请手动修改脚本中的 load_fasta_references 函数。")

    for cid, votes in cluster_votes.items():
        if not votes: continue
        most_common_tag = max(votes, key=votes.get)
        
        # 尝试直接匹配
        if most_common_tag in ref_dict:
            cluster_ref_seqs[cid] = ref_dict[most_common_tag]
        # 尝试去掉可能存在的 > 符号再匹配（以防万一）
        elif most_common_tag.replace(">", "") in ref_dict:
            cluster_ref_seqs[cid] = ref_dict[most_common_tag.replace(">", "")]
            
    del cluster_votes
    del ref_dict
    gc.collect()
    
    # 3.3 生成临时排序文件
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
                # 只有找到了对应 Reference 的簇才会被写入
                if cid != -1 and cid in cluster_ref_seqs:
                    seq = parts[-1]
                    fout.write(f"{cid}\t{seq}\n")
            valid_idx += 1
            
            if valid_idx % 1000000 == 0:
                print(f"      已预处理 {valid_idx} 条...", end='\r')

    del cluster_map
    gc.collect()

    # 3.4 外部排序
    print("\n   [3.4] 执行外部排序 (Linux Sort)...")
    # -n 按数值排, -S 50% 使用50%内存
    sort_cmd = f"sort -n -k1,1 -S 50% {temp_unsorted} -o {temp_sorted}"
    subprocess.run(sort_cmd, shell=True, check=True)
    
    os.remove(temp_unsorted)

    # 3.5 输出最终结果
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
            
            if cid != current_cid:
                if current_cid is not None:
                    fr.write("===============================\n")
                
                current_cid = cid
                ref_seq = cluster_ref_seqs[cid]
                ff.write(ref_seq + "\n")
                fr.write(ref_seq + "\n")
            
            fr.write(seq + "\n")
            
        if current_cid is not None:
            fr.write("===============================\n")

    if os.path.exists(temp_sorted):
        os.remove(temp_sorted)

    print("-" * 40)
    print(f"🎉 Goldman 数据处理完成！")
    print(f"👉 结果位置: {dir_feddna}")
    print("-" * 40)

if __name__ == "__main__":
    process_goldman_data()