import os

# ================= 配置 =================
# 输入文件 (原始脏数据)
INPUT_FILE = "给师妹的clover数据集/id20/id20_tags_reads.txt"
# 输出文件 (清洗后的数据)
OUTPUT_FILE = "给师妹的clover数据集/id20_tags_reads_clean.txt"

# 阈值：如果一条 read 中 N 的数量超过这个值，就丢弃
# 建议：如果是 150bp，N 超过 3-5 个其实就很难救了，这里设为 5 (3%)
MAX_N_COUNT = 5 
# =======================================

def clean_data():
    print(f"🧹 开始清洗数据: {os.path.basename(INPUT_FILE)}")
    print(f"   过滤标准: N count > {MAX_N_COUNT}")
    
    total_reads = 0
    kept_reads = 0
    n_counts = {}  # 统计 N 的分布

    with open(INPUT_FILE, 'r') as fin, open(OUTPUT_FILE, 'w') as fout:
        for line in fin:
            total_reads += 1
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            # id20 格式: Tag Sequence
            seq = parts[-1]
            
            # 计算 N 的数量 (大小写都算)
            n_count = seq.count('N') + seq.count('n')
            
            # 统计分布
            n_counts[n_count] = n_counts.get(n_count, 0) + 1
            
            if n_count <= MAX_N_COUNT:
                fout.write(line)
                kept_reads += 1
            
            if total_reads % 1000000 == 0:
                print(f"   已扫描 {total_reads} 条...", end='\r')

    print(f"\n✅ 清洗完成！")
    print(f"   原始总数: {total_reads}")
    print(f"   保留总数: {kept_reads} ({kept_reads/total_reads*100:.2f}%)")
    print(f"   丢弃总数: {total_reads - kept_reads}")
    
    print("\n📊 [N] 分布统计 (Top 10):")
    sorted_counts = sorted(n_counts.items(), key=lambda x: x[0])
    for count, freq in sorted_counts[:15]:
        print(f"   含有 {count} 个 N: {freq} 条 reads")
    
    if kept_reads == 0:
        print("\n❌ 警告: 所有 reads 都被丢弃了！请检查 MAX_N_COUNT 阈值。")
    else:
        print(f"\n👉 新文件位置: {OUTPUT_FILE}")
        print("💡 请修改 run_real_data_id20_Chunked.py 读取这个新文件，然后重跑 Step 0。")

if __name__ == "__main__":
    clean_data()