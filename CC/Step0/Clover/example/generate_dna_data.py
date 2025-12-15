import random

def generate_dna_with_variation(base_sequence, mutation_rate=0.05):
    """
    根据基础DNA序列生成带有随机突变的新序列。
    
    Args:
        base_sequence (str): 原始DNA序列（A、T、G、C组成）
        mutation_rate (float): 每个位点突变的概率，默认5%
    
    Returns:
        str: 新生成的变异DNA序列
    """
    bases = ['A', 'T', 'G', 'C']
    new_sequence = []
    
    for base in base_sequence:
        if random.random() < mutation_rate:
            # 变异成除自身以外的其他碱基
            new_base = random.choice([b for b in bases if b != base])
            new_sequence.append(new_base)
        else:
            new_sequence.append(base)
            
    return ''.join(new_sequence)

def generate_clustered_sequences(num_clusters=10, seqs_per_cluster=20, seq_length=152):
    """
    生成具有簇结构的DNA序列数据及其真实标签。
    
    Args:
        num_clusters (int): 簇的数量
        seqs_per_cluster (int): 每个簇包含的序列数量
        seq_length (int): 每条序列的长度
    
    Returns:
        (list of str, list of int): 生成的DNA序列列表 和 对应的真实簇标签列表
    """
    all_sequences = []
    true_labels = []
    
    for cluster_id in range(num_clusters):
        # 随机生成该簇的中心序列
        base_seq = ''.join(random.choice(['A', 'T', 'G', 'C']) for _ in range(seq_length))
        
        for _ in range(seqs_per_cluster):
            # 让突变率在1%-10%之间随机变化，增加多样性
            mut_rate = random.uniform(0.01, 0.10)
            seq = generate_dna_with_variation(base_seq, mut_rate)
            all_sequences.append(seq)
            true_labels.append(cluster_id)
    
    return all_sequences, true_labels

# 演示生成数据并保存到文件（改成txt格式）
if __name__ == "__main__":
    num_clusters = 10
    seqs_per_cluster = 20
    seq_length = 152
    
    sequences, labels = generate_clustered_sequences(num_clusters, seqs_per_cluster, seq_length)
    
    # 保存序列到txt文件，格式为: seq_id <tab> sequence
    with open("simulated_sequences.txt", "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f"seq{i}_cluster{labels[i]}\t{seq}\n")
    
    # 保存真实标签，方便评估聚类准确率
    with open("true_labels.txt", "w") as f:
        for i, label in enumerate(labels):
            f.write(f"seq{i}\t{label}\n")
    
    print(f"已生成{num_clusters}个簇，每簇{seqs_per_cluster}条序列，总计{len(sequences)}条序列")
    print("序列文件：simulated_sequences.txt")
    print("标签文件：true_labels.txt")
