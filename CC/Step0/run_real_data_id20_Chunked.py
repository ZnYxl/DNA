import os
import sys
import subprocess
import glob
import shutil
import array
from collections import defaultdict

# ================= 实验配置 =================
DATASET_NAME = "id20"
SOURCE_DIR = "给师妹的clover数据集/id20"
# ID20 的有效载荷长度 (去引物后)
# 原长 150 - 引物约 40 = 110
CLOVER_SEQ_LENGTH = 110 
CHUNK_SIZE = 5000000 
CLOVER_PROCESSES = 0 

# === 关键：引物锚点 (用于定位 Payload) ===
# Forward Primer 结尾: ...AGTGCAACAAG [TCAATCCG] -> Payload
ANCHOR_FWD = "TCAATCCG" 
# Payload 截取长度 (从锚点后开始取多少bp)
PAYLOAD_EXTRACT_LEN = 115 

# ===========================================

def load_fasta_references(fasta_path):
    print(f"📖 [Ref] 读取参考序列: {os.path.basename(fasta_path)} ...")
    refs = {}
    with open(fasta_path, 'r') as f:
        # 读取所有行
        lines = [l.strip() for l in f if l.strip()]
    
    # 自动检测格式
    if lines[0].startswith(">"):
        # 标准 FASTA
        current_tag = None
        for line in lines:
            if line.startswith(">"):
                current_tag = line[1:]
            elif current_tag:
                refs[current_tag] = line
    else:
        # ID20 特殊格式: Line 1 = ID, Line 2 = Seq
        print("   ℹ️ 检测到无 '>' 格式，启用双行读取模式...")
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                tag = lines[i]
                seq = lines[i+1]
                refs[tag] = seq
                
    print(f"   ✅ 加载了 {len(refs)} 条参考序列 (Ground Truth)")
    return refs

def extract_payload(sequence):
    """
    智能提取有效载荷：
    1. 搜索 Fwd Primer 的锚点
    2. 提取锚点后的序列作为 Payload
    3. 如果找不到锚点，返回原序列(截断)
    """
    pos = sequence.find(ANCHOR_FWD)
    if pos != -1:
        # 找到了锚点，取锚点之后的内容
        start = pos + len(ANCHOR_FWD)
        # 提取并截断/补齐到固定长度
        payload = sequence[start : start + CLOVER_SEQ_LENGTH]
    else:
        # 没找到锚点(可能是头部缺失)，直接取前段
        payload = sequence[:CLOVER_SEQ_LENGTH]
    
    # 长度对齐：如果短了就补N，长了已截断
    if len(payload) < CLOVER_SEQ_LENGTH:
        payload = payload.ljust(CLOVER_SEQ_LENGTH, 'N')
        
    return payload

def process_id20_fixed():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "Experiments", f"{DATASET_NAME}_Real")
    
    dir_raw = os.path.join(exp_dir, "01_FormattedInput")
    dir_clover = os.path.join(exp_dir, "02_CloverOut")
    dir_feddna = os.path.join(exp_dir, "03_FedDNA_In")
    dir_temp = os.path.join(exp_dir, "99_Temp")
    dir_chunks = os.path.join(exp_dir, "00_Chunks")
    
    # 清理旧的切片，防止混淆
    if os.path.exists(dir_chunks):
        shutil.rmtree(dir_chunks)
    
    for d in [dir_raw, dir_clover, dir_feddna, dir_temp, dir_chunks]:
        os.makedirs(d, exist_ok=True)

    # 优先读取清洗后的数据（如果存在），否则读取原始数据
    clean_reads_file = "id20_tags_reads_clean.txt"
    raw_reads_file = "id20_tags_reads.txt"
    
    if os.path.exists(os.path.join(SOURCE_DIR, clean_reads_file)):
        print(f"✨ 使用清洗后的数据: {clean_reads_file}")
        src_reads_path = os.path.join(SOURCE_DIR, clean_reads_file)
    else:
        print(f"⚠️ 未找到清洗数据，使用原始数据: {raw_reads_file}")
        src_reads_path = os.path.join(SOURCE_DIR, raw_reads_file)

    src_ref_path = os.path.join(SOURCE_DIR, "id20_refs.fasta")
    
    # === Step 1: 智能切片与去引物 ===
    print(f"\n[Step 1] 切片并提取 Payload (去除公共引物)...")
    if not os.path.exists(src_reads_path):
        print(f"❌ 错误: 找不到源文件 {src_reads_path}")
        return

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
                # === 核心修改: 提取 Payload ===
                clean_seq = extract_payload(raw_seq)
                # ===========================
                global_line_idx = line_count + 1
                current_out.write(f"{global_line_idx} {clean_seq}\n")
            
            line_count += 1
            
    if current_out: current_out.close()
    print(f"\n   ✅ 切片完成。共 {line_count} 条 reads。")
    existing_chunks = sorted(glob.glob(os.path.join(dir_chunks, "chunk_*.txt")))

    # === Step 2: 逐块运行 Clover ===
    print(f"\n[Step 2] 运行 Clover (L={CLOVER_SEQ_LENGTH})...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(current_dir, "Clover") + os.pathsep + env.get("PYTHONPATH", "")
    
    final_clover_result = os.path.join(dir_clover, "clover_result_merged.txt")
    
    # 为了保证结果正确，建议每次重跑 Step 2
    if os.path.exists(final_clover_result):
        os.remove(final_clover_result)
        
    with open(final_clover_result, 'w') as f_merged:
        pass 
        
    for i, chunk_path in enumerate(existing_chunks):
        chunk_name = os.path.basename(chunk_path)
        chunk_out_base = os.path.join(dir_chunks, f"out_{chunk_name}")
        chunk_out_txt = chunk_out_base + ".txt"
        
        print(f"   🚀 [{i+1}/{len(existing_chunks)}] 处理切片: {chunk_name}")
        
        # 注意: 这里使用提取后的 payload 长度 (110)
        cmd = [sys.executable, "-m", "clover.main", "-I", chunk_path, 
               "-O", chunk_out_base, "-L", str(CLOVER_SEQ_LENGTH), "-P", str(CLOVER_PROCESSES), "--no-tag"]
        try:
            subprocess.run(cmd, check=True, env=env) 
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 切片 {chunk_name} 处理失败！Exit Code: {e.returncode}")
            return
        
        # 合并
        with open(chunk_out_txt, 'r') as f_in, open(final_clover_result, 'a') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(chunk_out_txt)
        
    print(f"   ✅ Clover 聚类完成。")

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
    except StopIteration:
        pass

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
            if valid_idx % 5000000 == 0: print(f"      已投票 {valid_idx} 条...", end='\r')

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
            
    print(f"      成功匹配 Reference: {matched_count} (目标: ~60万)")

    del cluster_votes, ref_dict
    
    print("   [3.3] 生成排序中间文件...")
    temp_unsorted = os.path.join(dir_temp, "unsorted_reads.txt")
    temp_sorted = os.path.join(dir_temp, "sorted_reads.txt")
    
    # 注意：这里写入的是原始完整 reads (src_reads_path)，而不是 payload
    # 因为 FedDNA 训练通常需要完整的 reads (含引物)
    with open(src_reads_path, 'r') as fin, open(temp_unsorted, 'w') as fout:
        valid_idx = 0
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2: continue
            if valid_idx < len(cluster_map):
                cid = cluster_map[valid_idx]
                if cid != -1 and cid in cluster_ref_seqs:
                    fout.write(f"{cid}\t{parts[-1]}\n")
            valid_idx += 1
            if valid_idx % 5000000 == 0: print(f"      已预处理 {valid_idx} 条...", end='\r')
    
    del cluster_map
    print("\n   [3.4] 外部排序...")
    subprocess.run(f"sort -n -k1,1 -S 50% {temp_unsorted} -o {temp_sorted}", shell=True, check=True)
    os.remove(temp_unsorted)

    print("   [3.5] 写入最终文件...")
    out_read = os.path.join(dir_feddna, "read.txt")
    out_ref = os.path.join(dir_feddna, "ref.txt")
    
    current_cid = None
    cluster_count = 0
    with open(temp_sorted, 'r') as fin, open(out_read, 'w') as fr, open(out_ref, 'w') as ff:
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) != 2: continue
            cid = int(parts[0])
            seq = parts[1]
            if cid != current_cid:
                if current_cid is not None: fr.write("===============================\n")
                current_cid = cid
                cluster_count += 1
                ref_seq = cluster_ref_seqs[cid]
                ff.write(ref_seq + "\n")
                fr.write(ref_seq + "\n")
            fr.write(seq + "\n")
        if current_cid is not None: fr.write("===============================\n")
    
    if os.path.exists(temp_sorted): os.remove(temp_sorted)
    print(f"\n🎉 id20 修复版处理完毕！")
    print(f"📊 最终有效簇: {cluster_count} (应该接近 60万)")

if __name__ == "__main__":
    process_id20_fixed()