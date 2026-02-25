import os
import multiprocessing
from multiprocessing import Pool

try:
    import edlib
    HAS_EDLIB = True
    print("✅ 检测到 edlib，将使用高精度模糊比对去除引物 (Fuzzy Trimming)")
except ImportError:
    HAS_EDLIB = False
    print("⚠️ 未检测到 edlib，将使用固定长度 [25:-25] 裁剪 (Hard Trimming)")

# 定义两侧的引物序列
FWD = "TCCTGTGCTGCCTGTAATGAGCCAA"
REV = "AGCATAGAACTGAGACCACGGATTG"

def trim_seq(seq):
    seq = seq.strip()
    if not seq: return ""
    
    # 🌟 修复核心：绝对不要动分隔符！
    if seq.startswith("====="):
        return seq
        
    # 针对完美的 Reference，直接精准切割
    if len(seq) == 150 and seq.startswith(FWD) and seq.endswith(REV):
        return seq[25:-25]
        
    if HAS_EDLIB:
        res_fwd = edlib.align(FWD, seq[:35], mode="HW", task="locations")
        start = res_fwd['locations'][0][1] + 1 if res_fwd['locations'] else 25
        
        res_rev = edlib.align(REV, seq[-35:], mode="HW", task="locations")
        end = len(seq) - 35 + res_rev['locations'][-1][0] if res_rev['locations'] else len(seq) - 25
        
        start = min(max(start, 20), 30)
        end = max(min(end, len(seq) - 30), len(seq) - 20)
        
        return seq[start:end]
    else:
        return seq[25:-25]

def process_txt(in_path, out_path):
    if not os.path.exists(in_path):
        print(f"⚠️ 找不到文件: {in_path}")
        return
        
    print(f"⏳ 正在处理 {in_path} ...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(in_path, 'r') as f:
        lines = f.readlines()
        
    with Pool(multiprocessing.cpu_count()) as pool:
        trimmed = pool.map(trim_seq, lines)
        
    with open(out_path, 'w') as f:
        for line in trimmed:
            if line: # 过滤掉可能的空行
                f.write(line + "\n")
    print(f"✅ 已保存至 {out_path} (共 {len(trimmed):,} 行)")

def process_fasta(in_path, out_path):
    if not os.path.exists(in_path):
        print(f"⚠️ 找不到文件: {in_path}")
        return
        
    print(f"⏳ 正在处理 FASTA: {in_path} ...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    count = 0
    with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
        for line in fin:
            if line.startswith(">"):
                fout.write(line)
            else:
                fout.write(trim_seq(line) + "\n")
                count += 1
    print(f"✅ FASTA 已保存至 {out_path} (共 {count:,} 条 Reference)")

if __name__ == "__main__":
    # 原数据路径
    OLD_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_Real_last/03_FedDNA_In"
    OLD_FASTA = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1/exp1_refs.fasta"
    
    # 新数据路径
    NEW_DIR = "/mnt/st_data/liangxinyi/code/CC/Step0/Experiments/exp_1_NoPrimer/03_FedDNA_In"
    NEW_FASTA = "/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1_NoPrimer/exp1_refs.fasta"
    
    print("=" * 60)
    print("🧬 DNA Storage Primer Removal Pipeline 启动")
    print("=" * 60)
    
    process_txt(f"{OLD_DIR}/read.txt", f"{NEW_DIR}/read.txt")
    process_txt(f"{OLD_DIR}/ref.txt", f"{NEW_DIR}/ref.txt")
    process_fasta(OLD_FASTA, NEW_FASTA)
    
    print("=" * 60)
    print("🎉 全部处理完成！")