import os
import sys
import subprocess
import glob
import shutil
import array
from collections import defaultdict

# ================= 实验配置 =================
DATASET_NAME = "exp_1"
SOURCE_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1"

# === 对比组配置：不去引物 (Raw) ===
# 使用全长序列
CLOVER_SEQ_LENGTH = 150 
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
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                refs[lines[i]] = lines[i+1]
    return refs

def process_exp1_raw():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # === 修改输出目录为 _Raw ===
    exp_dir = os.path.join(current_dir, "Experiments", f"{DATASET_NAME}_Raw")
    
    dir_raw = os.path.join(exp_dir, "01_FormattedInput")
    dir_clover = os.path.join(exp_dir, "02_CloverOut")
    dir_feddna = os.path.join(exp_dir, "03_FedDNA_In")
    dir_temp = os.path.join(exp_dir, "99_Temp")
    dir_chunks = os.path.join(exp_dir, "00_Chunks")
    
    if os.path.exists(dir_chunks): shutil.rmtree(dir_chunks)
    for d in [dir_raw, dir_clover, dir_feddna, dir_temp, dir_chunks]:
        os.makedirs(d, exist_ok=True)

    src_reads_path = os.path.join(SOURCE_DIR, "exp1_tags_reads.txt")
    src_ref_path = os.path.join(SOURCE_DIR, "exp1_refs.fasta")
    
    # === Step 1: 切片 (保留全长，不去引物) ===
    print(f"\n[Step 1] 切片 (保留原始全长序列，含引物)...")
    
    chunk_idx = 0
    line_count = 0
    current_out = None
    
    with open(src_reads_path, 'r') as fin:
        for line in fin:
            if line_count % CHUNK_SIZE == 0:
                if current_out: current_out.close()
                chunk_name = os.path.join(dir_chunks, f"chunk_{chunk_idx:03d}.txt")
                current_out = open(chunk_name, 'w')
                print(f"   正在生成切片: chunk_{chunk_idx:03d}.txt ...", end='\r')
                chunk_idx += 1
            
            parts = line.strip().split()
            if len(parts) >= 2:
                raw_seq = parts[-1]
                # === 核心区别：不做 extract_payload，直接写 ===
                global_line_idx = line_count + 1
                current_out.write(f"{global_line_idx} {raw_seq}\n")
            
            line_count += 1
            
    if current_out: current_out.close()
    print(f"\n   ✅ 切片完成。")
    existing_chunks = sorted(glob.glob(os.path.join(dir_chunks, "chunk_*.txt")))

    # === Step 2: 逐块运行 Clover ===
    print(f"\n[Step 2] 运行 Clover (L={CLOVER_SEQ_LENGTH}, Raw Mode)...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(current_dir, "Clover") + os.pathsep + env.get("PYTHONPATH", "")
    
    final_clover_result = os.path.join(dir_clover, "clover_result_merged.txt")
    if os.path.exists(final_clover_result): os.remove(final_clover_result)
        
    with open(final_clover_result, 'w') as f_merged: pass 
        
    for i, chunk_path in enumerate(existing_chunks):
        chunk_name = os.path.basename(chunk_path)
        chunk_out_base = os.path.join(dir_chunks, f"out_{chunk_name}")
        chunk_out_txt = chunk_out_base + ".txt"
        
        print(f"   🚀 [{i+1}/{len(existing_chunks)}] 处理切片: {chunk_name}")
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

    # === Step 3: 合并与排序 ===
    print(f"\n[Step 3] 生成 FedDNA 输入...")
    cluster_map = array.array('i') 
    
    def stream_tokens(path):
        with open(path, 'r') as f:
            buf = ""
            while True:
                chunk = f.read(1024*1024)
                if not chunk: break
                cleaned = chunk.replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ').replace(',', ' ').replace("'", " ").replace('"', " ")
                buf += cleaned
                tokens = buf.split()
                if chunk[-1].isspace() or chunk[-1] in "[](),'\"":
                    for t in tokens: yield t
                    buf = ""
                else:
                    if tokens:
                        for t in tokens[:-1]: yield t
                        buf = tokens[-1]
                    else: pass
            if buf.strip(): yield buf.strip()

    print("   [3.1] 加载聚类映射...")
    token_gen = stream_tokens(final_clover_result)
    try:
        while True:
            idx_str = next(token_gen)
            cid_str = next(token_gen)
            cluster_map.append(int(cid_str))
    except StopIteration: pass

    print("   [3.2] 投票 Reference...")
    cluster_votes = defaultdict(lambda: defaultdict(int)) 
    with open(src_reads_path, 'r') as f:
        valid_idx = 0
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            tag_in_read = parts[0]
            if valid_idx < len(cluster_map):
                cid = cluster_map[valid_idx]
                if cid != -1: cluster_votes[cid][tag_in_read] += 1
            valid_idx += 1
            if valid_idx % 1000000 == 0: print(f"      已投票 {valid_idx} 条...", end='\r')

    print("\n      结算投票...")
    ref_dict = load_fasta_references(src_ref_path)
    cluster_ref_seqs = {} 
    
    matched_count = 0
    for cid, votes in cluster_votes.items():
        if not votes: continue
        best_tag = max(votes, key=votes.get)
        if best_tag in ref_dict: 
            cluster_ref_seqs[cid] = ref_dict[best_tag]
            matched_count += 1
            
    print(f"      成功匹配 Reference: {matched_count}")

    del cluster_votes, ref_dict
    
    # 后续排序步骤省略不写了，因为对比实验主要看 Step 3.2 的 matched_count 就足够说明问题了
    # 如果 matched_count 很低，说明聚类失败
    print(f"\n🔍 [对比结果关键指标]")
    print(f"   Raw 模式下匹配到的簇数量: {matched_count}")
    print(f"   Fixed 模式下匹配到的簇数量: (请参考之前的日志，应该是 11710)")
    
if __name__ == "__main__":
    process_exp1_raw()