import os
import sys
import subprocess
import glob
import shutil
from collections import defaultdict

# ================= 实验配置 =================
DATASET_NAME = "Goldman"
SOURCE_DIR = "给师妹的clover数据集"
SEQ_LENGTH = 117
CHUNK_SIZE = 5000000  
CLOVER_PROCESSES = 0  

# ===========================================

def load_fasta_references(fasta_path):
    print(f"📖 [Ref] 读取参考序列: {os.path.basename(fasta_path)} ...")
    refs = {}
    current_tag = None
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                current_tag = line[1:]
            else:
                if current_tag:
                    refs[current_tag] = line
    return refs

def process_goldman_chunked():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "Experiments", f"{DATASET_NAME}_Real")
    
    dir_raw = os.path.join(exp_dir, "01_FormattedInput")
    dir_clover = os.path.join(exp_dir, "02_CloverOut")
    dir_feddna = os.path.join(exp_dir, "03_FedDNA_In")
    dir_temp = os.path.join(exp_dir, "99_Temp")
    dir_chunks = os.path.join(exp_dir, "00_Chunks")
    
    for d in [dir_raw, dir_clover, dir_feddna, dir_temp, dir_chunks]:
        os.makedirs(d, exist_ok=True)

    reads_file = "now_goldman_tags_reads.txt"
    ref_file = "Goldman_fa.fasta"
    src_reads_path = os.path.join(SOURCE_DIR, reads_file)
    src_ref_path = os.path.join(SOURCE_DIR, ref_file)
    
    # === Step 1: 切片 (同前) ===
    existing_chunks = sorted(glob.glob(os.path.join(dir_chunks, "chunk_*.txt")))
    if not existing_chunks:
        print(f"\n[Step 1] 正在将大文件切分为 {CHUNK_SIZE} 条/块...")
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
                global_line_idx = line_count + 1
                parts = line.strip().split()
                if len(parts) >= 2:
                    current_out.write(f"{global_line_idx} {parts[-1]}\n")
                line_count += 1
        if current_out: current_out.close()
        print(f"\n   ✅ 切分完成！共 {line_count} 条。")
        existing_chunks = sorted(glob.glob(os.path.join(dir_chunks, "chunk_*.txt")))
    else:
        print(f"\n[Step 1] 检测到 {len(existing_chunks)} 个已有切片，跳过切分。")

    # === Step 2: 逐块运行 (Verbose Mode) ===
    print(f"\n[Step 2] 开始逐块运行 Clover (Verbose)...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(current_dir, "Clover") + os.pathsep + env.get("PYTHONPATH", "")
    
    final_clover_result = os.path.join(dir_clover, "clover_result_merged.txt")
    
    if not os.path.exists(final_clover_result):
        with open(final_clover_result, 'w') as f_merged:
            pass 
            
        for i, chunk_path in enumerate(existing_chunks):
            chunk_name = os.path.basename(chunk_path)
            chunk_out_base = os.path.join(dir_chunks, f"out_{chunk_name}")
            chunk_out_txt = chunk_out_base + ".txt"
            
            print(f"   🚀 [{i+1}/{len(existing_chunks)}] 处理切片: {chunk_name}")
            
            if not os.path.exists(chunk_out_txt) or os.path.getsize(chunk_out_txt) == 0:
                # 【修正】：移除了 stdout/stderr 的屏蔽，让 subprocess 直接输出到屏幕
                cmd = [sys.executable, "-m", "clover.main", "-I", chunk_path, 
                       "-O", chunk_out_base, "-L", str(SEQ_LENGTH), "-P", str(CLOVER_PROCESSES), "--no-tag"]
                try:
                    subprocess.run(cmd, check=True, env=env) # 这里不再吞掉输出了
                except subprocess.CalledProcessError as e:
                    print(f"\n❌ 切片 {chunk_name} 处理失败！")
                    print(f"   退出代码: {e.returncode}")
                    return
            else:
                print(f"      (已存在，跳过)")
            
            print(f"      正在合并结果...")
            with open(chunk_out_txt, 'r') as f_in, open(final_clover_result, 'a') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(chunk_out_txt)
            
        print(f"   ✅ 所有切片处理完毕。")
    else:
         print(f"   ✅ 检测到合并结果已存在，跳过 Step 2。")

    # === Step 3: 合并与排序 (同前) ===
    print(f"\n[Step 3] 生成 FedDNA 输入...")
    import array
    cluster_map = array.array('i') 
    
    def stream_tokens(path):
        with open(path, 'r') as f:
            buf = ""
            while True:
                chunk = f.read(1024*1024)
                if not chunk: break
                cleaned = chunk.replace('[', ' ').replace(']', ' ')\
                               .replace('(', ' ').replace(')', ' ')\
                               .replace(',', ' ').replace("'", " ").replace('"', " ")
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

    if os.path.exists(final_clover_result):
        token_gen = stream_tokens(final_clover_result)
        try:
            while True:
                idx_str = next(token_gen)
                cid_str = next(token_gen)
                cluster_map.append(int(cid_str))
        except StopIteration:
            pass
        print(f"      映射加载完毕，共 {len(cluster_map)} 条。")

        print("   [3.2] 投票 Reference...")
        cluster_votes = defaultdict(lambda: defaultdict(int)) 
        with open(src_reads_path, 'r') as f:
            valid_idx = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                if valid_idx < len(cluster_map):
                    cid = cluster_map[valid_idx]
                    if cid != -1: cluster_votes[cid][parts[0]] += 1
                valid_idx += 1
                if valid_idx % 5000000 == 0: print(f"      已投票 {valid_idx} 条...", end='\r')

        print("\n      结算投票...")
        ref_dict = load_fasta_references(src_ref_path)
        cluster_ref_seqs = {} 
        for cid, votes in cluster_votes.items():
            if not votes: continue
            best_tag = max(votes, key=votes.get)
            if best_tag in ref_dict: cluster_ref_seqs[cid] = ref_dict[best_tag]
            elif best_tag.replace(">", "") in ref_dict: cluster_ref_seqs[cid] = ref_dict[best_tag.replace(">", "")]

        del cluster_votes, ref_dict
        
        print("   [3.3] 生成排序中间文件...")
        temp_unsorted = os.path.join(dir_temp, "unsorted_reads.txt")
        temp_sorted = os.path.join(dir_temp, "sorted_reads.txt")
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
        
        del cluster_map
        print("\n   [3.4] 外部排序...")
        subprocess.run(f"sort -n -k1,1 -S 50% {temp_unsorted} -o {temp_sorted}", shell=True, check=True)
        os.remove(temp_unsorted)

        print("   [3.5] 写入最终文件...")
        out_read = os.path.join(dir_feddna, "read.txt")
        out_ref = os.path.join(dir_feddna, "ref.txt")
        current_cid = None
        with open(temp_sorted, 'r') as fin, open(out_read, 'w') as fr, open(out_ref, 'w') as ff:
            for line in fin:
                parts = line.strip().split('\t')
                if len(parts) != 2: continue
                cid = int(parts[0])
                if cid != current_cid:
                    if current_cid is not None: fr.write("===============================\n")
                    current_cid = cid
                    ref_seq = cluster_ref_seqs[cid]
                    ff.write(ref_seq + "\n")
                    fr.write(ref_seq + "\n")
                fr.write(parts[1] + "\n")
            if current_cid is not None: fr.write("===============================\n")
        
        if os.path.exists(temp_sorted): os.remove(temp_sorted)
        print(f"\n🎉 大功告成！")

if __name__ == "__main__":
    process_goldman_chunked()