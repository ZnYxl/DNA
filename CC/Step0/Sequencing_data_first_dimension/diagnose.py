import collections

def diagnose_output(input_file):
    print(f"🔍 正在深度诊断 {input_file} ...")
    
    total_lines = 0
    exact_duplicates = 0
    len_counts = collections.Counter()
    
    seen_lines = set()
    
    with open(input_file, 'r') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            total_lines += 1
            
            # 1. 统计完全重复的行 (Tag 和 Read 完全一致)
            if line in seen_lines:
                exact_duplicates += 1
            else:
                seen_lines.add(line)
                
            # 2. 统计长度分布
            parts = line.split('\t', 1)
            if len(parts) == 2:
                seq = parts[1]
                len_counts[len(seq)] += 1
                
    print("\n📊 诊断报告:")
    print(f"总读取行数: {total_lines}")
    print(f"完全重复的行数 (同一Tag+同一Read): {exact_duplicates}")
    
    # 计算去重后的理论剩余量
    unique_lines = total_lines - exact_duplicates
    print(f"👉 假如此刻执行完全去重，剩余: {unique_lines} (我们的目标是 2687556)")
    
    print("\n📏 序列长度分布 (Top 10):")
    for length, count in len_counts.most_common(10):
        print(f"  长度 {length}: {count} 条")

if __name__ == '__main__':
    # 确保文件名对应你当前的 output.txt
    diagnose_output('clover_input_first_dim.txt')