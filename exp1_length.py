import collections

def check_fasta_lengths(fasta_path):
    lengths = []
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    lengths.append(len("".join(current_seq)))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            lengths.append(len("".join(current_seq)))
            
    length_counts = collections.Counter(lengths)
    print("=== GT Reference 长度分布 ===")
    for length, count in sorted(length_counts.items()):
        print(f"长度 {length} bp: {count} 条序列")
    print(f"总计: {sum(length_counts.values())} 条序列")

# 替换为你实际的 GT 路径
check_fasta_lengths("/mnt/st_data/liangxinyi/code/CC/Step0/给师妹的clover数据集/exp_1/exp1_refs.fasta")