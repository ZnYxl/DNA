import os

# ================= 配置区域 =================
SRC_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1"
RAW_READS = os.path.join(SRC_DIR, "exp1_tags_reads.txt")
RAW_REFS = os.path.join(SRC_DIR, "exp1_refs.fasta")

# 输出目录
OUT_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_Final_Var/00_CleanData"

# 锚点 (Forward Primer 后半段)
ANCHOR_FWD = "GTAATGAGCCAA" 
PRIMER_FWD_LEN = 25
PRIMER_REV_LEN = 25

# ===========================================

def trim_ref(seq):
    """Reference 还是建议切掉两头引物，得到纯净的 Payload"""
    if len(seq) > (PRIMER_FWD_LEN + PRIMER_REV_LEN):
        return seq[PRIMER_FWD_LEN : -PRIMER_REV_LEN]
    return seq

def trim_read_variable(seq):
    """
    【柔性去引物】
    1. 找到锚点，切除头部引物（保证起点对齐）。
    2. 保留后面所有自然序列（允许变长）。
    """
    pos = seq.find(ANCHOR_FWD)
    if pos != -1:
        # 找到锚点，起点设为锚点之后
        start = pos + len(ANCHOR_FWD)
        # 取 start 之后的所有内容，不做截断！
        payload = seq[start:] 
    else:
        # 没找到锚点，为了保底，切掉前25bp
        if len(seq) > PRIMER_FWD_LEN:
            payload = seq[PRIMER_FWD_LEN:]
        else:
            payload = seq
            
    return payload

def run():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    clean_refs_path = os.path.join(OUT_DIR, "refs_clean.fasta")
    clean_reads_path = os.path.join(OUT_DIR, "reads_clean.txt")
    
    print(f"🚀 [Step 1] 开始柔性预处理 (Variable Length)...")
    
    # 1. 处理 Reference
    with open(RAW_REFS, 'r') as fin, open(clean_refs_path, 'w') as fout:
        lines = [l.strip() for l in fin if l.strip()]
        if lines[0].startswith(">"):
            for line in lines:
                if line.startswith(">"): fout.write(line + "\n")
                else: fout.write(trim_ref(line) + "\n")
        else:
            for i in range(0, len(lines), 2):
                fout.write(f"{lines[i]}\n{trim_ref(lines[i+1])}\n")
    
    # 2. 处理 Reads
    with open(RAW_READS, 'r') as fin, open(clean_reads_path, 'w') as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            clean_seq = trim_read_variable(parts[-1])
            fout.write(f"{parts[0]} {clean_seq}\n")
            
    print(f"\n✅ 预处理完成！Reads 起点已对齐，长度保持自然。")

if __name__ == "__main__":
    run()