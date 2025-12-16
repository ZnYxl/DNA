import random
import os
import ast
import collections
import numpy as np

# ==============================================================================
# 模块1: 高级数据生成 (Advanced Data Generation)
# ==============================================================================

def generate_diverse_references(num_clusters, seq_len, min_distance=0.3):
    """生成具有足够区分度的参考序列"""
    bases = ['A', 'C', 'G', 'T']
    references = []
    max_attempts = 1000
    
    # 1. 生成第一个
    first_ref = "".join(random.choice(bases) for _ in range(seq_len))
    references.append(first_ref)
    
    # 2. 生成其余的
    for i in range(1, num_clusters):
        attempts = 0
        while attempts < max_attempts:
            candidate = "".join(random.choice(bases) for _ in range(seq_len))
            
            # 计算最小汉明距离比例
            min_dist_ratio = float('inf')
            for existing_ref in references:
                dist = sum(c1 != c2 for c1, c2 in zip(candidate, existing_ref))
                ratio = dist / seq_len
                min_dist_ratio = min(min_dist_ratio, ratio)
            
            if min_dist_ratio >= min_distance:
                references.append(candidate)
                break
            attempts += 1
            
        if attempts == max_attempts:
            # 兜底策略：强制突变
            candidate = list(references[0])
            change_num = int(seq_len * min_distance) + 1
            positions = random.sample(range(seq_len), change_num)
            for pos in positions:
                original = candidate[pos]
                opts = [b for b in bases if b != original]
                candidate[pos] = random.choice(opts)
            references.append("".join(candidate))
            
    return references

def create_motif_based_references(num_clusters, seq_len):
    """基于motif的参考序列生成"""
    motifs = [
        "ATCGATCG", "GCGCGCGC", "ATATATATAT", "CGTACGTA", 
        "TGCATGCA", "AAGCTTAAGCTT", "GAATTCGAATTC", "GGATCCGGATCC"
    ]
    references = []
    for i in range(num_clusters):
        main_motif = motifs[i % len(motifs)]
        sequence = []
        pos = 0
        while pos < seq_len:
            if pos + len(main_motif) <= seq_len and random.random() < 0.6:
                sequence.extend(list(main_motif))
                pos += len(main_motif)
            else:
                sequence.append(random.choice(['A', 'C', 'G', 'T']))
                pos += 1
        ref_seq = "".join(sequence[:seq_len])
        # 补齐
        while len(ref_seq) < seq_len:
            ref_seq += random.choice(['A', 'C', 'G', 'T'])
        references.append(ref_seq)
    return references

def mutate_sequence_realistic(sequence, sub_rate=0.01, del_rate=0.005, ins_rate=0.005):
    """更真实的序列突变模型"""
    bases = ['A', 'C', 'G', 'T']
    result = list(sequence)
    i = 0
    while i < len(result):
        # 1. 替换 (Substitution)
        if random.random() < sub_rate:
            original = result[i]
            # 简单的转换偏好模拟
            if original in ['A', 'G']: # 嘌呤倾向
                target = 'G' if original == 'A' else 'A'
                result[i] = target if random.random() < 0.7 else random.choice(['C', 'T'])
            else: # 嘧啶倾向
                target = 'T' if original == 'C' else 'C'
                result[i] = target if random.random() < 0.7 else random.choice(['A', 'G'])
        
        # 2. 删除 (Deletion)
        if random.random() < del_rate:
            result.pop(i)
            continue # 删除后索引不变，但对应原序列下一位
            
        # 3. 插入 (Insertion)
        if random.random() < ins_rate:
            result.insert(i, random.choice(bases))
            i += 1 # 跳过刚插入的碱基
            
        i += 1
    return "".join(result)

# 【重点修改在这里！】注意函数定义里多了 reference_type 参数
def generate_data(output_dir, num_clusters=100, reads_per_cluster=20, seq_len=150, reference_type="diverse"):
    """
    主生成函数 (兼容 run_experiment.py 的调用接口)
    """
    raw_path = os.path.join(output_dir, "raw_reads.txt")
    gt_path = os.path.join(output_dir, "ground_truth.txt")
    
    print(f"🔧 [Advanced Generator] 生成配置: {num_clusters}簇, {reads_per_cluster}reads/簇, 模式={reference_type}")
    
    # 1. 生成参考序列
    if reference_type == "motif":
        ground_truths = create_motif_based_references(num_clusters, seq_len)
    else:
        ground_truths = generate_diverse_references(num_clusters, seq_len, min_distance=0.3)
        
    # 2. 生成 Reads
    all_reads_data = []
    counter = 0
    
    for cluster_idx, ref_seq in enumerate(ground_truths):
        for _ in range(reads_per_cluster):
            counter += 1
            read_id = str(counter)
            
            # 80% 概率高质量，20% 低质量
            if random.random() < 0.8:
                noisy_seq = mutate_sequence_realistic(ref_seq, sub_rate=0.005, del_rate=0.002, ins_rate=0.002)
                quality = "high"
            else:
                noisy_seq = mutate_sequence_realistic(ref_seq, sub_rate=0.02, del_rate=0.01, ins_rate=0.01)
                quality = "low"
                
            all_reads_data.append((read_id, noisy_seq, cluster_idx, ref_seq, quality))
            
    random.shuffle(all_reads_data)
    
    # 3. 写入文件
    with open(raw_path, 'w') as f:
        for item in all_reads_data:
            # 格式: ID [TAB] Sequence
            f.write(f"{item[0]}\t{item[1]}\n")
            
    with open(gt_path, 'w') as f:
        f.write("Read_ID\tCluster_ID\tRef_Seq\tQuality\n")
        for item in all_reads_data:
            f.write(f"{item[0]}\t{item[2]}\t{item[3]}\t{item[4]}\n")
            
    print(f"✅ 数据生成完毕: {len(all_reads_data)} 条 Reads")
    return raw_path, gt_path

# ==============================================================================
# 模块2: 格式转换 (Bridge Logic - 保持不变)
# ==============================================================================

def load_raw_reads(file_path):
    d = {}
    with open(file_path, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: d[p[0]] = p[1]
    return d

def clover_to_feddna(clover_out_path, raw_reads_path, output_dir):
    """将Clover结果转换为FedDNA格式"""
    raw_reads = load_raw_reads(raw_reads_path)
    clusters = collections.defaultdict(list)
    
    # 智能解析 Clover 输出
    try:
        with open(clover_out_path, 'r') as f:
            content = f.read().strip()
            if content.startswith("[") and content.endswith("]"):
                # 列表格式解析
                pairs = ast.literal_eval(content)
                for item in pairs:
                    if str(item[1]) not in ['-1', -1]:
                        clusters[str(item[1])].append(str(item[0]))
            else:
                # 逐行格式解析
                f.seek(0)
                for line in f:
                    p = line.replace(',', ' ').split()
                    if len(p) >= 2 and p[1] not in ['-1', '-1']:
                        clusters[p[1]].append(p[0])
    except Exception as e:
        print(f"❌ 解析 Clover 输出失败: {e}")
        return 0, ""

    # 写入结果
    out_read = os.path.join(output_dir, "read.txt")
    out_ref = os.path.join(output_dir, "ref.txt") # 统一叫 ref.txt
    
    valid_count = 0
    with open(out_read, 'w') as fr, open(out_ref, 'w') as ff:
        for cid, mems in clusters.items():
            if cid in raw_reads:
                # Center 作为伪 Reference
                ff.write(raw_reads[cid] + "\n") 
                for m in mems:
                    if m in raw_reads:
                        fr.write(raw_reads[m] + "\n")
                fr.write("===============================\n")
                valid_count += 1
                
    return valid_count, out_read